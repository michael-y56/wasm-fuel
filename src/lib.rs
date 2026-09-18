//! See the crate README for what this is. `binary` holds the module format
//! parser (built up section by section); `leb` is the integer encoding it
//! reads immediates with. This is the crate root: it assembles the section
//! readers into the one public entry point, [`parse`].

#![forbid(unsafe_code)]

pub mod binary;
pub mod leb;

use binary::{
    read_code_section, read_custom_section, read_export_section, read_function_section,
    read_header, read_import_section, read_start_section, read_type_section, skip_section,
    Export, FuncType, Import, LocalDecl, ParseError, ParseErrorKind,
};

/// One locally defined function: the type it was declared with in the
/// function section, and the locals and body decoded for it in the code
/// section. The format stores those two halves in separate sections so a
/// decoder can find a function's signature without reading any code; this
/// struct puts them back together once both sections are in hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Func {
    pub type_index: u32,
    pub locals: Vec<LocalDecl>,
    pub body: Vec<u8>,
}

/// A fully parsed module: every section this crate understands, assembled
/// into one place rather than left as the sequence of raw section reads that
/// produced them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub types: Vec<FuncType>,
    pub imports: Vec<Import>,
    pub funcs: Vec<Func>,
    pub exports: Vec<Export>,
    pub start: Option<u32>,
    /// Names of the custom sections found while parsing, in the order they
    /// appeared. A custom section may appear any number of times, anywhere
    /// between the sections listed above, without affecting their required
    /// relative order.
    pub custom_sections: Vec<String>,
    /// Ids of the table, memory, global, element, data and data-count
    /// sections that were present but skipped rather than decoded, in the
    /// order they appeared.
    pub skipped_sections: Vec<u8>,
}

fn peek_id(bytes: &[u8], pos: usize) -> Option<u8> {
    bytes.get(pos).copied()
}

/// Parses a complete module: the header, then each section in the order the
/// format requires - type, import, function, table, memory, global, export,
/// start, element, data count, code, data. Every section is optional in the
/// sense that a module need not use it, but a section that shows up before an
/// earlier-ordered one, or a second copy of a section that may only appear
/// once, is rejected with [`ParseErrorKind::SectionOutOfOrder`] at the offset
/// of its id byte. Table, memory, global, element, data and data-count
/// sections are skipped by length rather than decoded, since nothing
/// downstream of this crate needs their contents; their ids land in
/// [`Module::skipped_sections`]. Custom sections are exempt from the ordering
/// rule - any number of them may appear at any point in the byte stream - and
/// only their names, not their contents, land in [`Module::custom_sections`].
pub fn parse(bytes: &[u8]) -> Result<Module, ParseError> {
    read_header(bytes)?;
    let mut pos = 8;

    let mut custom_sections = Vec::new();
    let mut skipped_sections = Vec::new();
    let mut types = Vec::new();
    let mut imports = Vec::new();
    let mut type_indices = Vec::new();
    let mut exports = Vec::new();
    let mut start = None;
    let mut code = Vec::new();
    let mut code_seen = false;

    // The relative order a section id is required to appear in - distinct
    // from the id itself, since the data count section's id (12) is higher
    // than the code section's (10) even though it must come first.
    let mut last_order = 0u8;

    while let Some(id) = peek_id(bytes, pos) {
        if id == binary::SECTION_ID_CUSTOM {
            custom_sections.push(read_custom_section(bytes, &mut pos)?);
            continue;
        }

        let order = match id {
            binary::SECTION_ID_TYPE => 1,
            binary::SECTION_ID_IMPORT => 2,
            binary::SECTION_ID_FUNCTION => 3,
            binary::SECTION_ID_TABLE => 4,
            binary::SECTION_ID_MEMORY => 5,
            binary::SECTION_ID_GLOBAL => 6,
            binary::SECTION_ID_EXPORT => 7,
            binary::SECTION_ID_START => 8,
            binary::SECTION_ID_ELEMENT => 9,
            binary::SECTION_ID_DATA_COUNT => 10,
            binary::SECTION_ID_CODE => 11,
            binary::SECTION_ID_DATA => 12,
            _ => break,
        };
        if order <= last_order {
            return Err(ParseError { offset: pos, kind: ParseErrorKind::SectionOutOfOrder });
        }
        last_order = order;

        match id {
            binary::SECTION_ID_TYPE => types = read_type_section(bytes, &mut pos)?,
            binary::SECTION_ID_IMPORT => imports = read_import_section(bytes, &mut pos, types.len())?,
            binary::SECTION_ID_FUNCTION => {
                type_indices = read_function_section(bytes, &mut pos, types.len())?
            }
            binary::SECTION_ID_TABLE | binary::SECTION_ID_MEMORY | binary::SECTION_ID_GLOBAL => {
                skipped_sections.push(skip_section(bytes, &mut pos)?)
            }
            binary::SECTION_ID_EXPORT => exports = read_export_section(bytes, &mut pos)?,
            binary::SECTION_ID_START => start = Some(read_start_section(bytes, &mut pos)?),
            binary::SECTION_ID_ELEMENT | binary::SECTION_ID_DATA_COUNT => {
                skipped_sections.push(skip_section(bytes, &mut pos)?)
            }
            binary::SECTION_ID_CODE => {
                code = read_code_section(bytes, &mut pos, type_indices.len())?;
                code_seen = true;
            }
            binary::SECTION_ID_DATA => skipped_sections.push(skip_section(bytes, &mut pos)?),
            _ => unreachable!("every id reaching here matched the order table above"),
        }
    }

    // The code section is required exactly when the function section
    // declared at least one function - `read_code_section` itself checks the
    // count matches, but if the function section is non-empty and the code
    // section is absent entirely, that check never gets a chance to run.
    if !type_indices.is_empty() && !code_seen {
        return Err(ParseError { offset: pos, kind: ParseErrorKind::FunctionCodeMismatch });
    }

    if pos != bytes.len() {
        return Err(ParseError { offset: pos, kind: ParseErrorKind::UnknownSectionId });
    }

    let funcs = type_indices
        .into_iter()
        .zip(code)
        .map(|(type_index, c)| Func { type_index, locals: c.locals, body: c.body })
        .collect();

    Ok(Module { types, imports, funcs, exports, start, custom_sections, skipped_sections })
}

#[cfg(test)]
mod tests {
    use super::*;
    use binary::{ExportDesc, ImportDesc, ValType};

    // (module (func (export "square") (param i32) (result i32)
    //   local.get 0  local.get 0  i32.mul))
    const SQUARE: [u8; 41] = [
        0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x06, 0x01, 0x60, 0x01, 0x7F, 0x01, 0x7F,
        0x03, 0x02, 0x01, 0x00,
        0x07, 0x0A, 0x01, 0x06, 0x73, 0x71, 0x75, 0x61, 0x72, 0x65, 0x00, 0x00,
        0x0A, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x00, 0x6C, 0x0B,
    ];

    #[test]
    fn parses_the_readme_square_example() {
        let module = parse(&SQUARE).unwrap();
        assert_eq!(
            module.types,
            vec![FuncType { params: vec![ValType::I32], results: vec![ValType::I32] }]
        );
        assert_eq!(module.imports, vec![]);
        assert_eq!(
            module.funcs,
            vec![Func { type_index: 0, locals: vec![], body: vec![0x20, 0x00, 0x20, 0x00, 0x6C, 0x0B] }]
        );
        assert_eq!(
            module.exports,
            vec![Export { name: "square".to_string(), desc: ExportDesc::Func(0) }]
        );
        assert_eq!(module.start, None);
        assert_eq!(module.skipped_sections, vec![]);
    }

    #[test]
    fn parses_a_module_with_only_a_header() {
        let bytes = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        let module = parse(&bytes).unwrap();
        assert_eq!(module.types, vec![]);
        assert_eq!(module.imports, vec![]);
        assert_eq!(module.funcs, vec![]);
        assert_eq!(module.exports, vec![]);
        assert_eq!(module.start, None);
    }

    #[test]
    fn parses_an_imported_function_alongside_a_local_one() {
        // (import "env" "double" (func (param i32) (result i32)))
        // (func (export "run") (param i32) (result i32) local.get 0 end)
        let bytes = [
            0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x06, 0x01, 0x60, 0x01, 0x7F, 0x01, 0x7F, // type 0: (i32) -> i32
            0x02, 0x0E, 0x01, // import section, 1 import
            0x03, b'e', b'n', b'v', 0x06, b'd', b'o', b'u', b'b', b'l', b'e', 0x00, 0x00,
            0x03, 0x02, 0x01, 0x00, // function section: 1 func, type 0
            0x07, 0x07, 0x01, 0x03, b'r', b'u', b'n', 0x00, 0x01, // export "run" -> func 1
            0x0A, 0x06, 0x01, 0x04, 0x00, 0x20, 0x00, 0x0B, // code: no locals, local.get 0, end
        ];
        let module = parse(&bytes).unwrap();
        assert_eq!(
            module.imports,
            vec![binary::Import {
                module: "env".to_string(),
                name: "double".to_string(),
                desc: ImportDesc::Func(0),
            }]
        );
        assert_eq!(module.funcs, vec![Func { type_index: 0, locals: vec![], body: vec![0x20, 0x00, 0x0B] }]);
        assert_eq!(module.exports, vec![Export { name: "run".to_string(), desc: ExportDesc::Func(1) }]);
    }

    #[test]
    fn parses_a_module_with_skippable_sections_and_a_start_function() {
        let bytes = [
            0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
            0x04, 0x04, 0x01, 0x70, 0x00, 0x00, // table: funcref, limits {min:0}
            0x05, 0x03, 0x01, 0x00, 0x01, // memory: limits {min:1}
            0x08, 0x01, 0x00, // start: func 0
            0x0B, 0x02, 0x00, 0x00, // data: 1 entry, elided contents
        ];
        let module = parse(&bytes).unwrap();
        assert_eq!(module.start, Some(0));
        assert_eq!(
            module.skipped_sections,
            vec![binary::SECTION_ID_TABLE, binary::SECTION_ID_MEMORY, binary::SECTION_ID_DATA]
        );
    }

    #[test]
    fn collects_a_custom_section_name_before_the_type_section() {
        let mut bytes = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&[0x00, 0x05, 0x04, b'n', b'a', b'm', b'e']); // custom "name"
        let module = parse(&bytes).unwrap();
        assert_eq!(module.custom_sections, vec!["name".to_string()]);
        assert_eq!(module.skipped_sections, Vec::<u8>::new());
    }

    #[test]
    fn collects_custom_sections_interspersed_between_every_other_section() {
        let bytes = [
            0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x04, 0x03, b'p', b'r', b'e', // custom "pre", before the type section
            0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type 0: () -> ()
            0x00, 0x04, 0x03, b'm', b'i', b'd', // custom "mid", between type and function
            0x03, 0x02, 0x01, 0x00, // function section: 1 func, type 0
            0x00, 0x05, 0x04, b'p', b'o', b's', b't', // custom "post", after function
            0x0A, 0x04, 0x01, 0x02, 0x00, 0x0B, // code: no locals, end
        ];
        let module = parse(&bytes).unwrap();
        assert_eq!(
            module.custom_sections,
            vec!["pre".to_string(), "mid".to_string(), "post".to_string()]
        );
    }

    #[test]
    fn collects_a_run_of_consecutive_custom_sections() {
        let mut bytes = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&[0x00, 0x02, 0x01, b'a']); // custom "a"
        bytes.extend_from_slice(&[0x00, 0x02, 0x01, b'b']); // custom "b"
        let module = parse(&bytes).unwrap();
        assert_eq!(module.custom_sections, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn rejects_a_bad_header_before_looking_at_any_sections() {
        assert_eq!(
            parse(b"not wasm"),
            Err(ParseError { offset: 0, kind: ParseErrorKind::NotWasm })
        );
    }

    #[test]
    fn rejects_a_function_section_with_no_matching_code_section() {
        let bytes = [
            0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // type 0: () -> ()
            0x03, 0x02, 0x01, 0x00, // function section: 1 func, type 0, no code section follows
        ];
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 18, kind: ParseErrorKind::FunctionCodeMismatch })
        );
    }

    #[test]
    fn rejects_a_memory_section_before_the_table_section() {
        let mut bytes = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&[binary::SECTION_ID_MEMORY, 0x01, 0x00]); // memory, skipped
        bytes.extend_from_slice(&[binary::SECTION_ID_TABLE, 0x01, 0x00]); // table after memory - out of order
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 11, kind: ParseErrorKind::SectionOutOfOrder })
        );
    }

    #[test]
    fn rejects_a_duplicate_type_section() {
        let mut bytes = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&[0x01, 0x01, 0x00]); // type section, 0 types
        bytes.extend_from_slice(&[0x01, 0x01, 0x00]); // a second type section - not allowed
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 11, kind: ParseErrorKind::SectionOutOfOrder })
        );
    }

    #[test]
    fn rejects_an_export_section_after_the_start_section() {
        let mut bytes = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&[0x08, 0x01, 0x00]); // start: func 0
        bytes.extend_from_slice(&[0x07, 0x01, 0x00]); // export section, 0 exports - out of order
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 11, kind: ParseErrorKind::SectionOutOfOrder })
        );
    }

    #[test]
    fn rejects_trailing_bytes_that_are_not_a_recognized_section() {
        let mut bytes = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&[0x0D, 0x01, 0xAB]); // section id 13 - not a real section id
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 8, kind: ParseErrorKind::UnknownSectionId })
        );
    }
}

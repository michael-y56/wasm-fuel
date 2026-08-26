//! The WebAssembly binary format: module header first, sections after.
//!
//! This starts with just the header because everything downstream - section
//! framing, LEB128 sizes, the code section - is only meaningful once you know
//! the bytes are actually a wasm module and which version of the format they
//! claim to be. The header is fixed-width and not LEB128 encoded, unlike
//! nearly everything that follows it, which is why it gets its own read
//! logic instead of going through `leb`.

use crate::leb;
use std::fmt;

/// A WebAssembly module always opens with these four bytes.
const MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6D]; // "\0asm"

/// The only binary format version this crate understands.
const SUPPORTED_VERSION: u32 = 1;

/// The section id for the type section.
const SECTION_ID_TYPE: u8 = 1;

/// The section id for the import section.
const SECTION_ID_IMPORT: u8 = 2;

/// The section id for the function section.
const SECTION_ID_FUNCTION: u8 = 3;

/// The section id for the export section.
const SECTION_ID_EXPORT: u8 = 7;

/// The section id for the start section.
const SECTION_ID_START: u8 = 8;

/// The tag byte that opens every func type.
const FUNC_TYPE_TAG: u8 = 0x60;

/// Why parsing failed, and where.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseError {
    /// The byte offset that broke parsing.
    pub offset: usize,
    /// What went wrong there.
    pub kind: ParseErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// The first four bytes are not `\0asm`.
    NotWasm,
    /// The version field is not one this crate implements.
    UnsupportedVersion,
    /// The input ended before a required byte.
    UnexpectedEof,
    /// A LEB128 immediate did not decode; see [`crate::leb::LebError`].
    Leb,
    /// A section id this crate does not (yet) know how to read.
    UnknownSectionId,
    /// A section's declared size does not match the bytes it actually took
    /// to decode its contents.
    SectionSizeMismatch,
    /// A byte where a value type was expected is not one of `i32`/`i64`/
    /// `f32`/`f64`.
    InvalidValType,
    /// A func type did not open with the `0x60` tag.
    InvalidFuncType,
    /// An import's kind tag was not one of func/table/memory/global, or a
    /// tag-specific byte that follows it (a table's element type, a global's
    /// mutability flag) was not one of its valid values.
    InvalidExternKind,
    /// A table or memory's limits had a flag other than 0/1, or declared a
    /// maximum smaller than its minimum.
    InvalidLimits,
    /// A name (import module, import field) was not valid UTF-8.
    InvalidUtf8,
    /// A type index referenced by an import or a function was not within the
    /// bounds of the type section.
    TypeIndexOutOfRange,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self.kind {
            ParseErrorKind::NotWasm => "not a wasm module (bad magic number)",
            ParseErrorKind::UnsupportedVersion => "unsupported wasm version",
            ParseErrorKind::UnexpectedEof => "unexpected end of input",
            ParseErrorKind::Leb => "malformed LEB128 value",
            ParseErrorKind::UnknownSectionId => "unknown or unsupported section id",
            ParseErrorKind::SectionSizeMismatch => "section size does not match its contents",
            ParseErrorKind::InvalidValType => "not a valid value type",
            ParseErrorKind::InvalidFuncType => "func type missing its 0x60 tag",
            ParseErrorKind::InvalidExternKind => "not a valid import/export kind",
            ParseErrorKind::InvalidLimits => "invalid table or memory limits",
            ParseErrorKind::InvalidUtf8 => "name is not valid utf-8",
            ParseErrorKind::TypeIndexOutOfRange => "type index out of range",
        };
        write!(f, "{what} at offset {}", self.offset)
    }
}

impl std::error::Error for ParseError {}

/// Checks the eight-byte module header and returns the version it declares.
///
/// The version field is a plain little-endian `u32`, not LEB128 - the only
/// fixed-width multi-byte field in the whole format, because at this point
/// nothing has been negotiated yet that would let a decoder know how to read
/// anything else.
pub fn read_header(bytes: &[u8]) -> Result<u32, ParseError> {
    if bytes.len() < MAGIC.len() {
        return Err(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof });
    }
    if bytes[..MAGIC.len()] != MAGIC {
        return Err(ParseError { offset: 0, kind: ParseErrorKind::NotWasm });
    }
    if bytes.len() < MAGIC.len() + 4 {
        return Err(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof });
    }
    let version_bytes = [bytes[4], bytes[5], bytes[6], bytes[7]];
    let version = u32::from_le_bytes(version_bytes);
    if version != SUPPORTED_VERSION {
        return Err(ParseError { offset: 4, kind: ParseErrorKind::UnsupportedVersion });
    }
    Ok(version)
}

/// The four value types a signature can mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl ValType {
    fn from_byte(byte: u8) -> Option<ValType> {
        match byte {
            0x7F => Some(ValType::I32),
            0x7E => Some(ValType::I64),
            0x7D => Some(ValType::F32),
            0x7C => Some(ValType::F64),
            _ => None,
        }
    }
}

/// A function signature: some parameters, some results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

/// The lower and upper bound on a table or memory's size, in table
/// elements or 64 KiB pages respectively - the format does not distinguish
/// the two, so the unit is up to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub min: u32,
    pub max: Option<u32>,
}

/// What kind of thing an import binds, and the type-level details of each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportDesc {
    Func(u32),
    Table { element_type: u8, limits: Limits },
    Memory(Limits),
    Global { val_type: ValType, mutable: bool },
}

/// One entry of the import section: where it comes from and what it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    pub module: String,
    pub name: String,
    pub desc: ImportDesc,
}

/// What kind of thing an export binds, and the index into that kind's space.
/// Unlike an import, an export carries no type-level detail of its own - the
/// index is enough, since the thing it names was already fully described
/// wherever it was defined or imported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportDesc {
    Func(u32),
    Table(u32),
    Memory(u32),
    Global(u32),
}

/// One entry of the export section: the name other modules see, and what it
/// points to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub name: String,
    pub desc: ExportDesc,
}

fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8, ParseError> {
    let byte = *bytes
        .get(*pos)
        .ok_or(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })?;
    *pos += 1;
    Ok(byte)
}

/// Reads a section's id and its LEB128 content size. Does not read the
/// content itself, since what that content means depends on the id.
fn read_section_header(bytes: &[u8], pos: &mut usize) -> Result<(u8, u32), ParseError> {
    let id = read_u8(bytes, pos)?;
    let size_offset = *pos;
    let size = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: size_offset, kind: ParseErrorKind::Leb })?;
    Ok((id, size))
}

/// Reads a length-prefixed UTF-8 string, as used for import/export module
/// and field names.
fn read_name(bytes: &[u8], pos: &mut usize) -> Result<String, ParseError> {
    let len_offset = *pos;
    let len = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: len_offset, kind: ParseErrorKind::Leb })?;
    let start = *pos;
    let end = start
        .checked_add(len as usize)
        .filter(|&end| end <= bytes.len())
        .ok_or(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })?;
    let name = std::str::from_utf8(&bytes[start..end])
        .map_err(|_| ParseError { offset: start, kind: ParseErrorKind::InvalidUtf8 })?
        .to_string();
    *pos = end;
    Ok(name)
}

/// Reads a table or memory's limits: a flag byte, a minimum, and - if the
/// flag says so - a maximum that must not be smaller than the minimum.
fn read_limits(bytes: &[u8], pos: &mut usize) -> Result<Limits, ParseError> {
    let flag_offset = *pos;
    let flag = read_u8(bytes, pos)?;
    let min_offset = *pos;
    let min = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: min_offset, kind: ParseErrorKind::Leb })?;
    match flag {
        0x00 => Ok(Limits { min, max: None }),
        0x01 => {
            let max_offset = *pos;
            let max = leb::read_u32(bytes, pos)
                .map_err(|_| ParseError { offset: max_offset, kind: ParseErrorKind::Leb })?;
            if max < min {
                return Err(ParseError { offset: max_offset, kind: ParseErrorKind::InvalidLimits });
            }
            Ok(Limits { min, max: Some(max) })
        }
        _ => Err(ParseError { offset: flag_offset, kind: ParseErrorKind::InvalidLimits }),
    }
}

fn read_val_type_vec(bytes: &[u8], pos: &mut usize) -> Result<Vec<ValType>, ParseError> {
    let count_offset = *pos;
    let count = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: count_offset, kind: ParseErrorKind::Leb })?;
    // A malicious file can claim billions of entries in a few bytes; capping
    // the up-front allocation at the number of bytes actually left avoids
    // treating that claim at face value.
    let mut types = Vec::with_capacity(count.min(bytes.len() as u32) as usize);
    for _ in 0..count {
        let byte_offset = *pos;
        let byte = read_u8(bytes, pos)?;
        let val_type = ValType::from_byte(byte)
            .ok_or(ParseError { offset: byte_offset, kind: ParseErrorKind::InvalidValType })?;
        types.push(val_type);
    }
    Ok(types)
}

fn read_func_type(bytes: &[u8], pos: &mut usize) -> Result<FuncType, ParseError> {
    let tag_offset = *pos;
    let tag = read_u8(bytes, pos)?;
    if tag != FUNC_TYPE_TAG {
        return Err(ParseError { offset: tag_offset, kind: ParseErrorKind::InvalidFuncType });
    }
    let params = read_val_type_vec(bytes, pos)?;
    let results = read_val_type_vec(bytes, pos)?;
    Ok(FuncType { params, results })
}

/// Reads the type section: an id, a size, and a vector of func types.
///
/// `pos` must point at the section id. On success it is left just past the
/// section's declared size; on failure it is left wherever the byte that
/// broke parsing was.
pub fn read_type_section(bytes: &[u8], pos: &mut usize) -> Result<Vec<FuncType>, ParseError> {
    let header_offset = *pos;
    let (id, size) = read_section_header(bytes, pos)?;
    if id != SECTION_ID_TYPE {
        return Err(ParseError { offset: header_offset, kind: ParseErrorKind::UnknownSectionId });
    }

    let content_start = *pos;
    let content_end = content_start
        .checked_add(size as usize)
        .filter(|&end| end <= bytes.len())
        .ok_or(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })?;

    let count_offset = *pos;
    let count = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: count_offset, kind: ParseErrorKind::Leb })?;

    let mut types = Vec::with_capacity(count.min(bytes.len() as u32) as usize);
    for _ in 0..count {
        types.push(read_func_type(bytes, pos)?);
    }

    if *pos != content_end {
        return Err(ParseError { offset: *pos, kind: ParseErrorKind::SectionSizeMismatch });
    }
    Ok(types)
}

fn read_import(bytes: &[u8], pos: &mut usize, type_count: usize) -> Result<Import, ParseError> {
    let module = read_name(bytes, pos)?;
    let name = read_name(bytes, pos)?;
    let kind_offset = *pos;
    let kind = read_u8(bytes, pos)?;
    let desc = match kind {
        0x00 => {
            let idx_offset = *pos;
            let type_index = leb::read_u32(bytes, pos)
                .map_err(|_| ParseError { offset: idx_offset, kind: ParseErrorKind::Leb })?;
            if type_index as usize >= type_count {
                return Err(ParseError { offset: idx_offset, kind: ParseErrorKind::TypeIndexOutOfRange });
            }
            ImportDesc::Func(type_index)
        }
        0x01 => {
            let elem_offset = *pos;
            let element_type = read_u8(bytes, pos)?;
            if element_type != 0x70 {
                return Err(ParseError { offset: elem_offset, kind: ParseErrorKind::InvalidExternKind });
            }
            let limits = read_limits(bytes, pos)?;
            ImportDesc::Table { element_type, limits }
        }
        0x02 => ImportDesc::Memory(read_limits(bytes, pos)?),
        0x03 => {
            let vt_offset = *pos;
            let vt_byte = read_u8(bytes, pos)?;
            let val_type = ValType::from_byte(vt_byte)
                .ok_or(ParseError { offset: vt_offset, kind: ParseErrorKind::InvalidValType })?;
            let mut_offset = *pos;
            let mutable = match read_u8(bytes, pos)? {
                0x00 => false,
                0x01 => true,
                _ => return Err(ParseError { offset: mut_offset, kind: ParseErrorKind::InvalidExternKind }),
            };
            ImportDesc::Global { val_type, mutable }
        }
        _ => return Err(ParseError { offset: kind_offset, kind: ParseErrorKind::InvalidExternKind }),
    };
    Ok(Import { module, name, desc })
}

/// Reads the import section. `type_count` is the number of entries in the
/// type section, so a func import's type index can be range-checked as it is
/// read rather than left to fail later at call time.
pub fn read_import_section(
    bytes: &[u8],
    pos: &mut usize,
    type_count: usize,
) -> Result<Vec<Import>, ParseError> {
    let header_offset = *pos;
    let (id, size) = read_section_header(bytes, pos)?;
    if id != SECTION_ID_IMPORT {
        return Err(ParseError { offset: header_offset, kind: ParseErrorKind::UnknownSectionId });
    }

    let content_start = *pos;
    let content_end = content_start
        .checked_add(size as usize)
        .filter(|&end| end <= bytes.len())
        .ok_or(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })?;

    let count_offset = *pos;
    let count = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: count_offset, kind: ParseErrorKind::Leb })?;

    let mut imports = Vec::with_capacity(count.min(bytes.len() as u32) as usize);
    for _ in 0..count {
        imports.push(read_import(bytes, pos, type_count)?);
    }

    if *pos != content_end {
        return Err(ParseError { offset: *pos, kind: ParseErrorKind::SectionSizeMismatch });
    }
    Ok(imports)
}

/// Reads the function section: one type index per locally defined function,
/// in the order those functions will occupy the index space after the
/// functions brought in by imports.
pub fn read_function_section(
    bytes: &[u8],
    pos: &mut usize,
    type_count: usize,
) -> Result<Vec<u32>, ParseError> {
    let header_offset = *pos;
    let (id, size) = read_section_header(bytes, pos)?;
    if id != SECTION_ID_FUNCTION {
        return Err(ParseError { offset: header_offset, kind: ParseErrorKind::UnknownSectionId });
    }

    let content_start = *pos;
    let content_end = content_start
        .checked_add(size as usize)
        .filter(|&end| end <= bytes.len())
        .ok_or(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })?;

    let count_offset = *pos;
    let count = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: count_offset, kind: ParseErrorKind::Leb })?;

    let mut type_indices = Vec::with_capacity(count.min(bytes.len() as u32) as usize);
    for _ in 0..count {
        let idx_offset = *pos;
        let type_index = leb::read_u32(bytes, pos)
            .map_err(|_| ParseError { offset: idx_offset, kind: ParseErrorKind::Leb })?;
        if type_index as usize >= type_count {
            return Err(ParseError { offset: idx_offset, kind: ParseErrorKind::TypeIndexOutOfRange });
        }
        type_indices.push(type_index);
    }

    if *pos != content_end {
        return Err(ParseError { offset: *pos, kind: ParseErrorKind::SectionSizeMismatch });
    }
    Ok(type_indices)
}

fn read_export(bytes: &[u8], pos: &mut usize) -> Result<Export, ParseError> {
    let name = read_name(bytes, pos)?;
    let kind_offset = *pos;
    let kind = read_u8(bytes, pos)?;
    let idx_offset = *pos;
    let index = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: idx_offset, kind: ParseErrorKind::Leb })?;
    let desc = match kind {
        0x00 => ExportDesc::Func(index),
        0x01 => ExportDesc::Table(index),
        0x02 => ExportDesc::Memory(index),
        0x03 => ExportDesc::Global(index),
        _ => return Err(ParseError { offset: kind_offset, kind: ParseErrorKind::InvalidExternKind }),
    };
    Ok(Export { name, desc })
}

/// Reads the export section. Export indices are not range-checked here -
/// they name entries in the func/table/memory/global index spaces, which are
/// only fully known once imports and locally defined items are both
/// assembled, so that check happens where the whole module comes together.
pub fn read_export_section(bytes: &[u8], pos: &mut usize) -> Result<Vec<Export>, ParseError> {
    let header_offset = *pos;
    let (id, size) = read_section_header(bytes, pos)?;
    if id != SECTION_ID_EXPORT {
        return Err(ParseError { offset: header_offset, kind: ParseErrorKind::UnknownSectionId });
    }

    let content_start = *pos;
    let content_end = content_start
        .checked_add(size as usize)
        .filter(|&end| end <= bytes.len())
        .ok_or(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })?;

    let count_offset = *pos;
    let count = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: count_offset, kind: ParseErrorKind::Leb })?;

    let mut exports = Vec::with_capacity(count.min(bytes.len() as u32) as usize);
    for _ in 0..count {
        exports.push(read_export(bytes, pos)?);
    }

    if *pos != content_end {
        return Err(ParseError { offset: *pos, kind: ParseErrorKind::SectionSizeMismatch });
    }
    Ok(exports)
}

/// Reads the start section: a single function index, with no count prefix
/// since the section can name at most one function.
pub fn read_start_section(bytes: &[u8], pos: &mut usize) -> Result<u32, ParseError> {
    let header_offset = *pos;
    let (id, size) = read_section_header(bytes, pos)?;
    if id != SECTION_ID_START {
        return Err(ParseError { offset: header_offset, kind: ParseErrorKind::UnknownSectionId });
    }

    let content_start = *pos;
    let content_end = content_start
        .checked_add(size as usize)
        .filter(|&end| end <= bytes.len())
        .ok_or(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })?;

    let idx_offset = *pos;
    let index = leb::read_u32(bytes, pos)
        .map_err(|_| ParseError { offset: idx_offset, kind: ParseErrorKind::Leb })?;

    if *pos != content_end {
        return Err(ParseError { offset: *pos, kind: ParseErrorKind::SectionSizeMismatch });
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The header from the `square` module in the README example.
    const VALID: [u8; 8] = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];

    #[test]
    fn accepts_a_well_formed_header() {
        assert_eq!(read_header(&VALID), Ok(1));
    }

    #[test]
    fn accepts_trailing_bytes_after_the_header() {
        let mut bytes = VALID.to_vec();
        bytes.extend_from_slice(&[0x01, 0x00, 0x00, 0x00, 0x00]); // a section start
        assert_eq!(read_header(&bytes), Ok(1));
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut bytes = VALID;
        bytes[0] = 0x01;
        assert_eq!(
            read_header(&bytes),
            Err(ParseError { offset: 0, kind: ParseErrorKind::NotWasm })
        );
    }

    #[test]
    fn rejects_text_input() {
        assert_eq!(
            read_header(b"(module)"),
            Err(ParseError { offset: 0, kind: ParseErrorKind::NotWasm })
        );
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = VALID;
        bytes[4] = 0x02;
        assert_eq!(
            read_header(&bytes),
            Err(ParseError { offset: 4, kind: ParseErrorKind::UnsupportedVersion })
        );
    }

    #[test]
    fn rejects_truncated_magic() {
        assert_eq!(
            read_header(&VALID[..2]),
            Err(ParseError { offset: 2, kind: ParseErrorKind::UnexpectedEof })
        );
    }

    #[test]
    fn rejects_truncated_version() {
        assert_eq!(
            read_header(&VALID[..6]),
            Err(ParseError { offset: 6, kind: ParseErrorKind::UnexpectedEof })
        );
    }

    #[test]
    fn rejects_empty_input() {
        assert_eq!(
            read_header(&[]),
            Err(ParseError { offset: 0, kind: ParseErrorKind::UnexpectedEof })
        );
    }

    // (type (func (param i32 i32) (result i32)))
    const ONE_TYPE: [u8; 9] = [
        0x01, 0x07, // section id 1, size 7
        0x01, // 1 type
        0x60, 0x02, 0x7F, 0x7F, // func, 2 params: i32 i32
        0x01, 0x7F, // 1 result: i32
    ];

    #[test]
    fn reads_a_single_func_type() {
        let mut pos = 0;
        assert_eq!(
            read_type_section(&ONE_TYPE, &mut pos),
            Ok(vec![FuncType { params: vec![ValType::I32, ValType::I32], results: vec![ValType::I32] }])
        );
        assert_eq!(pos, ONE_TYPE.len());
    }

    #[test]
    fn reads_an_empty_type_section() {
        let bytes = [0x01, 0x01, 0x00]; // id 1, size 1, count 0
        let mut pos = 0;
        assert_eq!(read_type_section(&bytes, &mut pos), Ok(vec![]));
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn reads_more_than_one_type() {
        // type 0: () -> (), type 1: (i32) -> (i64)
        let bytes = [
            0x01, 0x09, // section id 1, size 9
            0x02, // 2 types
            0x60, 0x00, 0x00, // func, no params, no results
            0x60, 0x01, 0x7F, 0x01, 0x7E, // func, 1 param i32, 1 result i64
        ];
        let mut pos = 0;
        assert_eq!(
            read_type_section(&bytes, &mut pos),
            Ok(vec![
                FuncType { params: vec![], results: vec![] },
                FuncType { params: vec![ValType::I32], results: vec![ValType::I64] },
            ])
        );
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn rejects_a_section_id_that_is_not_type() {
        let mut bytes = ONE_TYPE;
        bytes[0] = 0x02; // import section id, not implemented yet
        let mut pos = 0;
        assert_eq!(
            read_type_section(&bytes, &mut pos),
            Err(ParseError { offset: 0, kind: ParseErrorKind::UnknownSectionId })
        );
    }

    #[test]
    fn rejects_a_func_type_missing_its_tag() {
        let bytes = [0x01, 0x04, 0x01, 0x61, 0x00, 0x00]; // tag 0x61 instead of 0x60
        let mut pos = 0;
        assert_eq!(
            read_type_section(&bytes, &mut pos),
            Err(ParseError { offset: 3, kind: ParseErrorKind::InvalidFuncType })
        );
    }

    #[test]
    fn rejects_an_invalid_val_type() {
        let bytes = [0x01, 0x05, 0x01, 0x60, 0x01, 0x7B, 0x00]; // 0x7B is not a value type
        let mut pos = 0;
        assert_eq!(
            read_type_section(&bytes, &mut pos),
            Err(ParseError { offset: 5, kind: ParseErrorKind::InvalidValType })
        );
    }

    #[test]
    fn rejects_a_declared_size_that_is_too_small() {
        let bytes = [0x01, 0x02, 0x00, 0xFF]; // size says 2 bytes, count only takes 1
        let mut pos = 0;
        assert_eq!(
            read_type_section(&bytes, &mut pos),
            Err(ParseError { offset: 3, kind: ParseErrorKind::SectionSizeMismatch })
        );
    }

    #[test]
    fn rejects_a_declared_size_that_overruns_the_input() {
        let bytes = [0x01, 0x05, 0x00]; // size says 5 bytes, only 1 remains
        let mut pos = 0;
        assert_eq!(
            read_type_section(&bytes, &mut pos),
            Err(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })
        );
    }

    #[test]
    fn rejects_a_truncated_count() {
        let bytes = [0x01, 0x01, 0x80]; // count byte has its continuation bit set with no follow-up
        let mut pos = 0;
        assert_eq!(
            read_type_section(&bytes, &mut pos),
            Err(ParseError { offset: 2, kind: ParseErrorKind::Leb })
        );
    }

    #[test]
    fn rejects_a_truncated_size() {
        let bytes = [0x01, 0x80]; // size byte has its continuation bit set with no follow-up
        let mut pos = 0;
        assert_eq!(
            read_type_section(&bytes, &mut pos),
            Err(ParseError { offset: 1, kind: ParseErrorKind::Leb })
        );
    }

    #[test]
    fn every_single_byte_corruption_of_one_type_is_an_error_not_a_panic() {
        for i in 0..ONE_TYPE.len() {
            for bad in [0x00u8, 0xFFu8] {
                let mut bytes = ONE_TYPE;
                if bytes[i] == bad {
                    continue;
                }
                bytes[i] = bad;
                let mut pos = 0;
                let _ = read_type_section(&bytes, &mut pos); // must not panic
            }
        }
    }

    #[test]
    fn every_truncation_of_one_type_is_an_error_not_a_panic() {
        for i in 0..ONE_TYPE.len() {
            let mut pos = 0;
            let result = read_type_section(&ONE_TYPE[..i], &mut pos);
            assert!(result.is_err(), "truncation to {i} bytes should not parse");
        }
    }

    // (import "env" "double" (func (param i32) (result i32)))
    const ONE_FUNC_IMPORT: [u8; 16] = [
        0x02, 0x0E, // section id 2, size 14
        0x01, // 1 import
        0x03, b'e', b'n', b'v', // module "env"
        0x06, b'd', b'o', b'u', b'b', b'l', b'e', // name "double"
        0x00, 0x00, // kind func, type index 0
    ];

    #[test]
    fn reads_a_single_func_import() {
        let mut pos = 0;
        assert_eq!(
            read_import_section(&ONE_FUNC_IMPORT, &mut pos, 1),
            Ok(vec![Import {
                module: "env".to_string(),
                name: "double".to_string(),
                desc: ImportDesc::Func(0),
            }])
        );
        assert_eq!(pos, ONE_FUNC_IMPORT.len());
    }

    #[test]
    fn reads_table_memory_and_global_imports() {
        let bytes = [
            0x02, 0x18, // section id 2, size 24
            0x03, // 3 imports
            0x01, b'w', 0x01, b't', // module "w", name "t"
            0x01, 0x70, 0x00, 0x01, // table: funcref, limits {min:1, max:none}
            0x01, b'w', 0x01, b'm', // module "w", name "m"
            0x02, 0x01, 0x01, 0x02, // memory: limits {min:1, max:2}
            0x01, b'w', 0x01, b'g', // module "w", name "g"
            0x03, 0x7F, 0x01, // global: i32, mutable
        ];
        let mut pos = 0;
        assert_eq!(
            read_import_section(&bytes, &mut pos, 0),
            Ok(vec![
                Import {
                    module: "w".to_string(),
                    name: "t".to_string(),
                    desc: ImportDesc::Table { element_type: 0x70, limits: Limits { min: 1, max: None } },
                },
                Import {
                    module: "w".to_string(),
                    name: "m".to_string(),
                    desc: ImportDesc::Memory(Limits { min: 1, max: Some(2) }),
                },
                Import {
                    module: "w".to_string(),
                    name: "g".to_string(),
                    desc: ImportDesc::Global { val_type: ValType::I32, mutable: true },
                },
            ])
        );
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn rejects_an_unknown_import_kind() {
        let mut bytes = ONE_FUNC_IMPORT;
        bytes[14] = 0x04; // no such extern kind
        let mut pos = 0;
        assert_eq!(
            read_import_section(&bytes, &mut pos, 1),
            Err(ParseError { offset: 14, kind: ParseErrorKind::InvalidExternKind })
        );
    }

    #[test]
    fn rejects_a_func_import_with_an_out_of_range_type_index() {
        let mut pos = 0;
        assert_eq!(
            read_import_section(&ONE_FUNC_IMPORT, &mut pos, 0), // no types declared
            Err(ParseError { offset: 15, kind: ParseErrorKind::TypeIndexOutOfRange })
        );
    }

    #[test]
    fn rejects_limits_with_a_max_below_the_min() {
        let bytes = [
            0x02, 0x09, // section id 2, size 9
            0x01, // 1 import
            0x01, b'w', 0x01, b'm', // module "w", name "m"
            0x02, 0x01, 0x05, 0x01, // memory: limits {min:5, max:1}
        ];
        let mut pos = 0;
        assert_eq!(
            read_import_section(&bytes, &mut pos, 0),
            Err(ParseError { offset: 10, kind: ParseErrorKind::InvalidLimits })
        );
    }

    #[test]
    fn rejects_a_name_that_is_not_utf8() {
        let bytes = [
            0x02, 0x07, // section id 2, size 7
            0x01, // 1 import
            0x01, 0xFF, // module: 1 byte, not valid utf-8
            0x01, b'x', // name "x"
            0x00, 0x00, // kind func, type index 0
        ];
        let mut pos = 0;
        assert_eq!(
            read_import_section(&bytes, &mut pos, 1),
            Err(ParseError { offset: 4, kind: ParseErrorKind::InvalidUtf8 })
        );
    }

    #[test]
    fn every_single_byte_corruption_of_one_func_import_is_an_error_not_a_panic() {
        for i in 0..ONE_FUNC_IMPORT.len() {
            for bad in [0x00u8, 0xFFu8] {
                let mut bytes = ONE_FUNC_IMPORT;
                if bytes[i] == bad {
                    continue;
                }
                bytes[i] = bad;
                let mut pos = 0;
                let _ = read_import_section(&bytes, &mut pos, 1); // must not panic
            }
        }
    }

    #[test]
    fn every_truncation_of_one_func_import_is_an_error_not_a_panic() {
        for i in 0..ONE_FUNC_IMPORT.len() {
            let mut pos = 0;
            let result = read_import_section(&ONE_FUNC_IMPORT[..i], &mut pos, 1);
            assert!(result.is_err(), "truncation to {i} bytes should not parse");
        }
    }

    #[test]
    fn reads_function_section_entries() {
        let bytes = [
            0x03, 0x03, // section id 3, size 3
            0x02, // 2 functions
            0x00, 0x01, // type indices 0 and 1
        ];
        let mut pos = 0;
        assert_eq!(read_function_section(&bytes, &mut pos, 2), Ok(vec![0, 1]));
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn reads_an_empty_function_section() {
        let bytes = [0x03, 0x01, 0x00]; // id 3, size 1, count 0
        let mut pos = 0;
        assert_eq!(read_function_section(&bytes, &mut pos, 0), Ok(vec![]));
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn rejects_a_function_section_type_index_out_of_range() {
        let bytes = [0x03, 0x02, 0x01, 0x00]; // 1 function, type index 0, but 0 types exist
        let mut pos = 0;
        assert_eq!(
            read_function_section(&bytes, &mut pos, 0),
            Err(ParseError { offset: 3, kind: ParseErrorKind::TypeIndexOutOfRange })
        );
    }

    #[test]
    fn rejects_a_function_section_id_that_is_not_function() {
        let bytes = [0x02, 0x01, 0x00]; // import section id, not function
        let mut pos = 0;
        assert_eq!(
            read_function_section(&bytes, &mut pos, 0),
            Err(ParseError { offset: 0, kind: ParseErrorKind::UnknownSectionId })
        );
    }

    #[test]
    fn every_truncation_of_two_funcs_is_an_error_not_a_panic() {
        let bytes = [0x03, 0x03, 0x02, 0x00, 0x01];
        for i in 0..bytes.len() {
            let mut pos = 0;
            let result = read_function_section(&bytes[..i], &mut pos, 2);
            assert!(result.is_err(), "truncation to {i} bytes should not parse");
        }
    }

    // (export "square" (func 0))
    const ONE_FUNC_EXPORT: [u8; 12] = [
        0x07, 0x0A, // section id 7, size 10
        0x01, // 1 export
        0x06, b's', b'q', b'u', b'a', b'r', b'e', // name "square"
        0x00, 0x00, // kind func, index 0
    ];

    #[test]
    fn reads_a_single_func_export() {
        let mut pos = 0;
        assert_eq!(
            read_export_section(&ONE_FUNC_EXPORT, &mut pos),
            Ok(vec![Export { name: "square".to_string(), desc: ExportDesc::Func(0) }])
        );
        assert_eq!(pos, ONE_FUNC_EXPORT.len());
    }

    #[test]
    fn reads_table_memory_and_global_exports() {
        let bytes = [
            0x07, 0x0D, // section id 7, size 13
            0x03, // 3 exports
            0x01, b't', 0x01, 0x00, // name "t", kind table, index 0
            0x01, b'm', 0x02, 0x00, // name "m", kind memory, index 0
            0x01, b'g', 0x03, 0x01, // name "g", kind global, index 1
        ];
        let mut pos = 0;
        assert_eq!(
            read_export_section(&bytes, &mut pos),
            Ok(vec![
                Export { name: "t".to_string(), desc: ExportDesc::Table(0) },
                Export { name: "m".to_string(), desc: ExportDesc::Memory(0) },
                Export { name: "g".to_string(), desc: ExportDesc::Global(1) },
            ])
        );
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn reads_an_empty_export_section() {
        let bytes = [0x07, 0x01, 0x00]; // id 7, size 1, count 0
        let mut pos = 0;
        assert_eq!(read_export_section(&bytes, &mut pos), Ok(vec![]));
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn rejects_an_unknown_export_kind() {
        let mut bytes = ONE_FUNC_EXPORT;
        bytes[10] = 0x04; // no such extern kind
        let mut pos = 0;
        assert_eq!(
            read_export_section(&bytes, &mut pos),
            Err(ParseError { offset: 10, kind: ParseErrorKind::InvalidExternKind })
        );
    }

    #[test]
    fn rejects_an_export_section_id_that_is_not_export() {
        let bytes = [0x03, 0x01, 0x00]; // function section id, not export
        let mut pos = 0;
        assert_eq!(
            read_export_section(&bytes, &mut pos),
            Err(ParseError { offset: 0, kind: ParseErrorKind::UnknownSectionId })
        );
    }

    #[test]
    fn every_single_byte_corruption_of_one_func_export_is_an_error_not_a_panic() {
        for i in 0..ONE_FUNC_EXPORT.len() {
            for bad in [0x00u8, 0xFFu8] {
                let mut bytes = ONE_FUNC_EXPORT;
                if bytes[i] == bad {
                    continue;
                }
                bytes[i] = bad;
                let mut pos = 0;
                let _ = read_export_section(&bytes, &mut pos); // must not panic
            }
        }
    }

    #[test]
    fn every_truncation_of_one_func_export_is_an_error_not_a_panic() {
        for i in 0..ONE_FUNC_EXPORT.len() {
            let mut pos = 0;
            let result = read_export_section(&ONE_FUNC_EXPORT[..i], &mut pos);
            assert!(result.is_err(), "truncation to {i} bytes should not parse");
        }
    }

    #[test]
    fn reads_a_start_section() {
        let bytes = [0x08, 0x01, 0x02]; // section id 8, size 1, function index 2
        let mut pos = 0;
        assert_eq!(read_start_section(&bytes, &mut pos), Ok(2));
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn rejects_a_start_section_id_that_is_not_start() {
        let bytes = [0x07, 0x01, 0x00]; // export section id, not start
        let mut pos = 0;
        assert_eq!(
            read_start_section(&bytes, &mut pos),
            Err(ParseError { offset: 0, kind: ParseErrorKind::UnknownSectionId })
        );
    }

    #[test]
    fn rejects_a_start_section_size_that_does_not_match_its_index() {
        let bytes = [0x08, 0x02, 0x00, 0x00]; // size says 2 bytes, index only takes 1
        let mut pos = 0;
        assert_eq!(
            read_start_section(&bytes, &mut pos),
            Err(ParseError { offset: 3, kind: ParseErrorKind::SectionSizeMismatch })
        );
    }

    #[test]
    fn every_truncation_of_a_start_section_is_an_error_not_a_panic() {
        let bytes = [0x08, 0x01, 0x02];
        for i in 0..bytes.len() {
            let mut pos = 0;
            let result = read_start_section(&bytes[..i], &mut pos);
            assert!(result.is_err(), "truncation to {i} bytes should not parse");
        }
    }
}

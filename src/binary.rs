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

/// The section id for the type section, the only one read so far.
const SECTION_ID_TYPE: u8 = 1;

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
}

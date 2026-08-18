//! The WebAssembly binary format: module header first, sections after.
//!
//! This starts with just the header because everything downstream - section
//! framing, LEB128 sizes, the code section - is only meaningful once you know
//! the bytes are actually a wasm module and which version of the format they
//! claim to be. The header is fixed-width and not LEB128 encoded, unlike
//! nearly everything that follows it, which is why it gets its own read
//! logic instead of going through `leb`.

use std::fmt;

/// A WebAssembly module always opens with these four bytes.
const MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6D]; // "\0asm"

/// The only binary format version this crate understands.
const SUPPORTED_VERSION: u32 = 1;

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
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self.kind {
            ParseErrorKind::NotWasm => "not a wasm module (bad magic number)",
            ParseErrorKind::UnsupportedVersion => "unsupported wasm version",
            ParseErrorKind::UnexpectedEof => "unexpected end of input",
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
}

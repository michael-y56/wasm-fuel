//! See the crate README for what this is. `binary` holds the module format
//! parser (built up section by section); `leb` is the integer encoding it
//! reads immediates with.

#![forbid(unsafe_code)]

pub mod binary;
pub mod leb;

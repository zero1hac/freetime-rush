
#![allow(unused_variables)]
fn main() {
// This structure cannot be printed either with `fmt::Display` or
// with `fmt::Debug`
#[allow(dead_code)]
struct UnPrintable(i32);

// The `derive` attribute automatically creates the implementation
// required to make this `struct` printable with `fmt::Debug`.
#[derive(Debug)]
#[allow(dead_code)]
struct DebugPrintable(i32);
}

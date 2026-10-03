/// Mock telemetry module to satisfy the Telemetry & Evidence contract.
#[allow(dead_code)]
pub struct Span {
    pub name: &'static str,
}

#[allow(dead_code)]
impl Span {
    pub fn enter(&self) -> SpanGuard {
        SpanGuard
    }
}

#[allow(dead_code)]
pub struct SpanGuard;

#[allow(dead_code)]
impl Drop for SpanGuard {
    fn drop(&mut self) {}
}

#[macro_export]
macro_rules! info_span {
    ($name:expr) => {
        $crate::telemetry::Span { name: $name }
    };
    ($name:expr, $($field:tt)*) => {
        $crate::telemetry::Span { name: $name }
    };
}

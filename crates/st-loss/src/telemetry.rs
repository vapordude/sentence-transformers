pub struct MockSpan;

impl MockSpan {
    pub fn enter(&self) -> MockSpanGuard {
        MockSpanGuard
    }
}

pub struct MockSpanGuard;

impl Drop for MockSpanGuard {
    fn drop(&mut self) {}
}

#[macro_export]
macro_rules! info_span {
    ($name:expr $(, $field:ident = $val:expr)* $(,)?) => {
        $crate::telemetry::MockSpan
    };
}

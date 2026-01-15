//! Event parsing and correlation from Nockchain logs
//!
//! This module provides tools for extracting events from logs and
//! correlating them with memory usage time-series data.

pub mod log_parser;

pub use log_parser::{
    LogEvent, EventType, LogParser, EventCorrelator, CorrelatedSample,
};

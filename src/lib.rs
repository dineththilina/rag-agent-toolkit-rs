// src/lib.rs
//
// Library surface of agent-toolkit. The binary (`src/main.rs`) is a thin server
// shell on top of these modules; exposing them as a library lets integration
// tests drive the real retrieval system as a black box.

pub mod agent;
pub mod api;
pub mod config;
pub mod metrics;
pub mod models;
pub mod rag;
pub mod sessions;

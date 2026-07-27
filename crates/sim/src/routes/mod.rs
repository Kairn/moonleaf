//! One module per route.

// Axum's `Handler` is implemented for functions returning a future, so a
// handler is `async` whether or not it awaits anything.
#![allow(clippy::unused_async)]

pub mod chat;
pub mod health;
pub mod models;

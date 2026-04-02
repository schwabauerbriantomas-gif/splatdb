pub mod wal;
pub mod metadata_store;
pub mod sqlite_store;
pub mod json_store;
pub mod persistence;

pub use metadata_store::{DocumentRecord, MetadataStore};
pub use persistence::{M2MPersistence, StorageBackend};

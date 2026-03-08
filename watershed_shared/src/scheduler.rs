use crate::basic_pooling::{get_tuple_vec, return_tuple_vec, CollectTupleVec};
use crate::caching::StrToKey;
use crate::global_logger::NO_AUX_DATA;
use crate::Tuple;
use core::f64;
use itertools::Itertools;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use tokio::sync::mpsc::error::{SendError as TokioSendError, TrySendError as TokioTrySendError};

#[derive(Debug, Clone)]
pub enum SyncPipe {
    Active(crossbeam::channel::Sender<Vec<Tuple>>),
    Dummy,
}

impl SyncPipe {
    pub fn send(
        &self,
        tuples: Vec<Tuple>,
    ) -> Result<(), crossbeam::channel::SendError<Vec<Tuple>>> {
        match self {
            SyncPipe::Active(tx) => tx.send(tuples),
            SyncPipe::Dummy => {
                for tuple in tuples {
                    let tuple_id = tuple.id();
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("failed to get time since epoch")
                        .as_nanos();
                    let diff = now - tuple.unix_time_created_ns;
                    debug!(
                        "dummy pipeline received tuple with id {tuple_id} with time diff {diff} ns"
                    );
                }
                Ok(())
            }
        }
    }
}

// #[derive(Debug, Clone)]
// pub enum AsyncPipe {
//     Active(tokio::sync::mpsc::UnboundedSender<Vec<Tuple>>),
//     // Active(tokio::sync::mpsc::Sender<Vec<Tuple>>),
//     Dummy,
// }

// impl AsyncPipe {
//     pub fn new() -> (Self, tokio::sync::mpsc::UnboundedReceiver<Vec<Tuple>>) {
//         let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
//         (AsyncPipe::Active(tx), rx)
//     }

//     pub fn dummy() -> Self {
//         AsyncPipe::Dummy
//     }
// }

#[derive(Debug, Clone)]
pub enum AsyncPipe {
    Active(BoundedAsyncSender),
    // Active(tokio::sync::mpsc::Sender<Vec<Tuple>>),
    Dummy,
}

pub fn bounded_channel(max_capacity: usize, max_age_ns: u128) -> (AsyncPipe, BoundedAsyncReceiver) {
    AsyncPipe::new_with_dummy(max_capacity, max_age_ns)
}
pub fn bounded_backup_channel(
    max_capacity: usize,
    max_age_ns: u128,
    backup_channel: Option<BackupChannel>,
) -> (AsyncPipe, BoundedAsyncReceiver) {
    if let Some(b) = backup_channel {
        AsyncPipe::new_with_backup(max_capacity, max_age_ns, b)
    } else {
        AsyncPipe::new_with_dummy(max_capacity, max_age_ns)
    }
}

impl AsyncPipe {
    pub fn new(max_capacity: usize, max_age_ns: u128) -> (Self, BoundedAsyncReceiver) {
        // Self::new_with_backup(max_capacity, max_age_ns, BackupChannel::NoOp)
        Self::new_with_backup(max_capacity, max_age_ns, BackupChannel::Dummy)
    }
    pub fn new_with_dummy(max_capacity: usize, max_age_ns: u128) -> (Self, BoundedAsyncReceiver) {
        Self::new_with_backup(max_capacity, max_age_ns, BackupChannel::Dummy)
    }
    pub fn new_with_backup(
        max_capacity: usize,
        max_age_ns: u128,
        backup_channel: BackupChannel,
    ) -> (Self, BoundedAsyncReceiver) {
        debug!("creating new async pipe with max capacity {max_capacity} and max age {max_age_ns}");
        let (tx, rx) = tokio::sync::mpsc::channel(max_capacity);
        (
            AsyncPipe::Active(BoundedAsyncSender {
                channel: tx,
                backup_channel: backup_channel.clone(),
            }),
            BoundedAsyncReceiver {
                max_age_ns,
                channel: rx,
                backup_channel,
            },
        )
    }
    pub fn dummy() -> Self {
        AsyncPipe::Dummy
    }

    pub fn len(&self) -> usize {
        match self {
            AsyncPipe::Active(bounded_async_sender) => bounded_async_sender.len(),
            AsyncPipe::Dummy => 0,
        }
    }
    pub fn cap(&self) -> usize {
        match self {
            AsyncPipe::Active(bounded_async_sender) => bounded_async_sender.cap(),
            AsyncPipe::Dummy => 0,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_full(&self) -> bool {
        self.len() == self.cap()
    }
}

#[derive(Debug, Clone)]
pub enum AsyncPipeSendError {
    Disconnected,
    Full,
}

#[derive(Debug, Clone)]
pub enum AsyncPipeRecvError {
    Disconnected,
    Empty,
}

#[derive(Debug, Clone)]
pub struct BoundedAsyncSender {
    pub(crate) channel: tokio::sync::mpsc::Sender<Vec<Tuple>>,
    pub(crate) backup_channel: BackupChannel,
}

// #[cfg(debug_assertions)]
impl Drop for BoundedAsyncSender {
    fn drop(&mut self) {
        // #[cfg(debug_assertions)]
        {
            if let Some(task) = tokio::task::try_id() {
                let thread = std::thread::current().id();
                debug!("dropping bounded async sender in task {task:?}, thread {thread:?}");
            } else {
                let thread = std::thread::current().id();
                debug!(
                    "dropping bounded async sender outside of a tokio runtime on thread {thread:?}"
                );
            }
        }
    }
}

impl BoundedAsyncSender {
    // unify the functions between the sender and receiver
    pub fn len(&self) -> usize {
        self.channel.max_capacity() - self.channel.capacity()
    }
    pub fn cap(&self) -> usize {
        self.channel.max_capacity()
    }
    pub fn remaining_capacity(&self) -> usize {
        self.channel.capacity()
    }

    pub fn extract_sender(self) -> tokio::sync::mpsc::Sender<Vec<Tuple>> {
        let this = std::mem::ManuallyDrop::new(self);
        this.channel.clone()
    }

    pub fn try_send_and_return(
        &self,
        tuples: Vec<Tuple>,
    ) -> Result<(), (Vec<Tuple>, Result<(), AsyncPipeSendError>)> {
        debug!(
            "inside bounded async sender try_send with stats len: {}, cap: {}, remaining: {}",
            self.len(),
            self.cap(),
            self.remaining_capacity()
        );
        match self.channel.try_send(tuples) {
            Ok(()) => {
                trace!("try_send success");
                Ok(())
            }
            Err(TokioTrySendError::Full(tuples)) => {
                warn!("channel was full, sending to backup");
                Err((tuples, Ok(())))
            }
            Err(TokioTrySendError::Closed(tuples)) => {
                warn!("channel was closed, dropping tuples");
                Err((tuples, Err(AsyncPipeSendError::Disconnected)))
            }
        }
    }

    pub fn try_send(&self, tuples: Vec<Tuple>) -> Result<(), AsyncPipeSendError> {
        debug!(
            "inside bounded async sender try_send with stats len: {}, cap: {}, remaining: {}",
            self.len(),
            self.cap(),
            self.remaining_capacity()
        );
        match self.channel.try_send(tuples) {
            Ok(()) => {
                debug!("try_send success");
                Ok(())
            }
            Err(TokioTrySendError::Full(tuples)) => {
                warn!("channel was full, sending to backup");
                match (&self.backup_channel, tuples.is_empty()) {
                    (BackupChannel::Unbounded(backup_channel), false) => {
                        if let Err(e) = backup_channel.send(tuples) {
                            error!("failed to send tuples to backup channel: {e}");
                        }
                    }
                    (BackupChannel::BoundedAndDrop(backup_channel), false) => {
                        if let Err(e) = backup_channel.try_send(tuples) {
                            error!("failed to send tuples to backup channel: {e}");
                        }
                    }
                    (BackupChannel::Dummy, false) => {
                        if let Err(e) = AsyncPipe::dummy().send(tuples) {
                            error!("failed to send tuples to dummy channel: {e:?}");
                        }
                    }
                    (
                        BackupChannel::Unbounded(..)
                        | BackupChannel::BoundedAndDrop(..)
                        | BackupChannel::Dummy,
                        true,
                    ) => {
                        return_tuple_vec(tuples);
                    }
                }
                Err(AsyncPipeSendError::Full)
            }
            Err(TokioTrySendError::Closed(tuples)) => {
                warn!("channel was closed, dropping tuples");
                match &self.backup_channel {
                    BackupChannel::Unbounded(backup_channel) => {
                        if let Err(e) = backup_channel.send(tuples) {
                            error!("failed to send tuples to backup channel: {e}");
                        }
                    }
                    BackupChannel::BoundedAndDrop(backup_channel) => {
                        if let Err(e) = backup_channel.try_send(tuples) {
                            error!("failed to send tuples to backup channel: {e}");
                        }
                    }
                    BackupChannel::Dummy => {
                        if let Err(e) = AsyncPipe::dummy().send(tuples) {
                            error!("failed to send tuples to dummy channel: {e:?}");
                        }
                    }
                }
                Err(AsyncPipeSendError::Disconnected)
            }
        }
    }

    pub async fn send(&self, tuples: Vec<Tuple>) -> Result<(), AsyncPipeSendError> {
        debug!(
            "inside bounded async sender async send with stats len: {}, cap: {}, remaining: {}",
            self.len(),
            self.cap(),
            self.remaining_capacity()
        );
        match self.channel.send(tuples).await {
            Ok(()) => {
                debug!("async send success");
                Ok(())
            }
            Err(TokioSendError(tuples)) => {
                match (&self.backup_channel, tuples.is_empty()) {
                    (BackupChannel::Unbounded(backup_channel), false) => {
                        if let Err(e) = backup_channel.send(tuples) {
                            error!("failed to send tuples to backup channel: {e}");
                        }
                    }
                    (BackupChannel::BoundedAndDrop(backup_channel), false) => {
                        if let Err(e) = backup_channel.try_send(tuples) {
                            error!("failed to send tuples to backup channel: {e}");
                        }
                    }
                    (BackupChannel::Dummy, false) => {
                        if let Err(e) = AsyncPipe::dummy().send(tuples) {
                            error!("failed to send tuples to dummy channel: {e:?}");
                        }
                    }
                    (
                        BackupChannel::Unbounded(..)
                        | BackupChannel::BoundedAndDrop(..)
                        | BackupChannel::Dummy,
                        true,
                    ) => {
                        return_tuple_vec(tuples);
                    }
                }
                Err(AsyncPipeSendError::Disconnected)
            }
        }
    }
}

#[derive(Debug)]
pub struct BoundedAsyncReceiver {
    pub(crate) max_age_ns: u128,
    pub(crate) channel: tokio::sync::mpsc::Receiver<Vec<Tuple>>,
    pub(crate) backup_channel: BackupChannel,
}

// #[cfg(debug_assertions)]
impl Drop for BoundedAsyncReceiver {
    fn drop(&mut self) {
        // #[cfg(debug_assertions)]
        {
            if let Some(task) = tokio::task::try_id() {
                let thread = std::thread::current().id();
                debug!("dropping bounded async receiver in task {task:?}, thread {thread:?}");
            } else {
                let thread = std::thread::current().id();
                debug!("dropping bounded async receiver outside of a runtime on thread {thread:?}");
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum BackupChannel {
    // NoOp,
    Dummy,
    BoundedAndDrop(tokio::sync::mpsc::Sender<Vec<Tuple>>),
    Unbounded(tokio::sync::mpsc::UnboundedSender<Vec<Tuple>>),
}

impl BoundedAsyncReceiver {
    // unify the functions between the sender and receiver
    pub fn len(&self) -> usize {
        self.channel.len()
    }
    pub fn cap(&self) -> usize {
        self.channel.max_capacity()
    }
    pub fn remaining_capacity(&self) -> usize {
        self.channel.max_capacity() - self.channel.len()
    }

    /// Receive a value, handling possible errors along the way. This will only return None if the channel is closed and empty.
    /// If the channel is closed but not empty, it will return the remaining values.
    /// Handled errors include
    /// - the channel being closed (returns None)
    /// - the channel being empty (waits for more data)
    /// - the channel received data, but it was all too old (waits for more data)
    ///
    /// Whenever the channel receives data that has been in the channel for too long, it will send the data to the backup channel if it exists.
    pub async fn recv(&mut self) -> Option<Vec<Tuple>> {
        self.recv_max_age(None).await
    }
    pub async fn recv_max_age(&mut self, max_age_ns: Option<u128>) -> Option<Vec<Tuple>> {
        const MINIMUM_BATCH_SIZE: usize = 4;
        debug!("async waiting to receive from channel");
        while let Some(mut v) = self.channel.recv().await {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("failed to get time since epoch")
                .as_nanos();
            let too_old_cond = |tuple: &Tuple| {
                let age = now - tuple.unix_time_created_ns();
                age > self.max_age_ns || max_age_ns.map_or(false, |m| age > m)
            };
            v.sort_by_key(&too_old_cond);
            let mut ages: smallvec::SmallVec<[(usize, f64); MINIMUM_BATCH_SIZE]> =
                smallvec::SmallVec::new();
            for tuple in v.iter() {
                let age = now - tuple.unix_time_created_ns();
                ages.push((tuple.id(), age as f64));
            }
            // let point = v.binary_search_by_key(&true, &too_old_cond);
            let point = v.iter().position(&too_old_cond);
            debug!(
                "async-recv g1 - received {} tuples, ids and ages: {:?}. discard point was found to be {:?}",
                v.len(),
                ages,
                point
            );
            let discard: Vec<_> = if let Some(point) = point {
                v.drain(point..).collect_tuple_vec()
            } else {
                get_tuple_vec()
            };
            if !discard.is_empty() {
                let drop_ids: smallvec::SmallVec<[_; MINIMUM_BATCH_SIZE]> =
                    discard.iter().map(|t| t.id()).collect();
                debug!(
                    "dropping {:?} tuples because they're too old, ids: {:?}",
                    discard.len(),
                    drop_ids
                );
            }
            match (&self.backup_channel, discard.is_empty()) {
                (BackupChannel::Unbounded(backup_channel), false) => {
                    debug!("async-recv g4-unbounded");
                    if let Err(e) = backup_channel.send(discard) {
                        error!("failed to send tuples to backup channel: {e}");
                    }
                }
                (BackupChannel::BoundedAndDrop(backup_channel), false) => {
                    debug!("async-recv g4-bounded");
                    if let Err(e) = backup_channel.try_send(discard) {
                        error!("failed to send tuples to backup channel: {e}");
                        if let Err(e) = AsyncPipe::Dummy.send(e.into_inner()) {
                            error!("failed to send tuples to dummy channel: {e:?}");
                        }
                    }
                }
                (BackupChannel::Dummy, false) => {
                    debug!("async-recv g4-dummy");
                    if let Err(e) = AsyncPipe::dummy().send(discard) {
                        error!("failed to send tuples to dummy channel: {e:?}");
                    }
                }
                (
                    BackupChannel::Unbounded(..)
                    | BackupChannel::BoundedAndDrop(..)
                    | BackupChannel::Dummy,
                    true,
                ) => {
                    debug!("async-recv g4-nothing to do");
                    return_tuple_vec(discard);
                }
            }
            debug!("async-recv g5 - {} items remaining", v.len());
            if !v.is_empty() {
                return Some(v);
            }
            return_tuple_vec(v);
        }
        None
    }

    pub fn try_recv(&mut self) -> Result<Option<Vec<Tuple>>, AsyncPipeRecvError> {
        debug!("trying to receive from channel");
        debug!("try_recv g1");
        match self.channel.try_recv() {
            Ok(mut v) => {
                const MINIMUM_BATCH_SIZE: usize = 4;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("failed to get time since epoch")
                    .as_nanos();
                let too_old_cond =
                    |tuple: &Tuple| now - tuple.unix_time_created_ns() > self.max_age_ns;
                v.sort_by_key(&too_old_cond);
                let mut ages: smallvec::SmallVec<[(usize, f64); MINIMUM_BATCH_SIZE]> =
                    smallvec::SmallVec::new();
                for tuple in v.iter() {
                    let age = now - tuple.unix_time_created_ns();
                    ages.push((tuple.id(), age as f64));
                }
                debug!(
                    "async-recv g1 - received {} tuples, ids and ages: {:?}",
                    v.len(),
                    ages
                );
                let point = v.binary_search_by_key(&true, &too_old_cond);
                let discard: Vec<_> = if let Ok(point) = point {
                    v.drain(point..).collect_tuple_vec()
                } else {
                    get_tuple_vec()
                };
                debug!("try_recv g2 - discard vec has {} elements", discard.len());
                if !discard.is_empty() {
                    debug!(
                        "dropping {:?} tuples because they're too old",
                        discard.len()
                    );
                }
                match (&self.backup_channel, discard.is_empty()) {
                    (BackupChannel::Unbounded(backup_channel), false) => {
                        debug!("try_recv g3-unbounded");
                        if let Err(e) = backup_channel.send(discard) {
                            error!("failed to send tuples to backup channel: {e}");
                        }
                    }
                    (BackupChannel::BoundedAndDrop(backup_channel), false) => {
                        debug!("try_recv g3-bounded");
                        if let Err(e) = backup_channel.try_send(discard) {
                            error!("failed to send tuples to backup channel: {e}");
                        }
                    }
                    (BackupChannel::Dummy, false) => {
                        debug!("try_recv g3-dummy");
                        if let Err(e) = AsyncPipe::dummy().send(discard) {
                            error!("failed to send tuples to dummy channel: {e:?}");
                        }
                    }
                    (
                        BackupChannel::Unbounded(..)
                        | BackupChannel::BoundedAndDrop(..)
                        | BackupChannel::Dummy,
                        true,
                    ) => {
                        debug!("try_recv g3-nothing to do");
                        return_tuple_vec(discard);
                    }
                }
                debug!("try_recv g4 - {} items remaining", v.len());
                if !v.is_empty() {
                    Ok(Some(v))
                } else {
                    // we filtered out everything, but that's not an error exactly
                    return_tuple_vec(v);
                    Ok(None)
                }
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => Err(AsyncPipeRecvError::Empty),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                Err(AsyncPipeRecvError::Disconnected)
            }
        }
    }
}

impl AsyncPipe {
    pub fn send(&self, tuples: Vec<Tuple>) -> Result<(), AsyncPipeSendError> {
        match self {
            AsyncPipe::Active(tx) => match tx.try_send(tuples) {
                Ok(()) => {
                    // debug!("active success! sent tuples to channel");
                    Ok(())
                }
                Err(AsyncPipeSendError::Full) => {
                    error!("channel was full, dropping tuples");
                    Err(AsyncPipeSendError::Full)
                }
                Err(AsyncPipeSendError::Disconnected) => {
                    error!("channel was closed, dropping tuples");
                    Err(AsyncPipeSendError::Disconnected)
                }
            },
            AsyncPipe::Dummy => {
                // debug!("how did we end up in a dummy?");
                let mut drop_string = Vec::new();
                for tuple in tuples {
                    #[allow(unused_labels)]
                    'log_dummy_drop: {
                        let log_location = "dummy_channel_drop";
                        let tuple_id = tuple.id();
                        let aux_data = NO_AUX_DATA;
                        if let Err(e) = crate::global_logger::log_data(
                            tuple_id,
                            log_location.to_raw_key(),
                            aux_data,
                        ) {
                            error!("failed to log dummy channel data: {e:?}");
                        }
                    }
                    drop_string.clear();
                    let tuple_id = tuple.id();
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("failed to get time since epoch")
                        .as_nanos();
                    let diff = now - tuple.unix_time_created_ns;
                    let mut write_failure = false;
                    if let Err(e) = serde_json::to_writer(&mut drop_string, &tuple) {
                        error!("failed to serialize tuple to string: {e}");
                        write_failure = true;
                    }
                    warn!(
                        "dummy pipeline received tuple with id {tuple_id} with time diff {diff} ns. dropping now {:?}",
                        if write_failure {
                            "<failed to serialize tuple>"
                        } else {
                            std::str::from_utf8(drop_string.as_slice()).unwrap_or("failed to convert to utf8")
                        }
                    );
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ShareableArray<'a, T, const N: usize = 3> {
    OwnedHeap(Vec<T>),
    OwnedInline([T; N]),
    Borrowed(&'a [T]),
    Shared(std::sync::Arc<[T]>),
}

impl<'a, T, const N: usize> ShareableArray<'a, T, N>
where
    T: Clone,
{
    pub fn as_mut(&mut self) -> &mut [T] {
        match self {
            ShareableArray::OwnedHeap(v) => v,
            ShareableArray::OwnedInline(v) => v,
            ShareableArray::Borrowed(v) => {
                *self = ShareableArray::OwnedHeap(v.to_vec());
                match self {
                    ShareableArray::OwnedHeap(v) => v,
                    _ => unreachable!(),
                }
            }
            ShareableArray::Shared(v) => {
                *self = ShareableArray::OwnedHeap(v.to_vec());
                match self {
                    ShareableArray::OwnedHeap(v) => v,
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl<'a, 'b, T, const N: usize> Deserialize<'b> for ShareableArray<'a, T, N>
where
    T: Deserialize<'b>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'b>,
    {
        let v = Vec::<T>::deserialize(deserializer)?;
        Ok(ShareableArray::OwnedHeap(v))
    }
}
impl<'a, T, const N: usize> Serialize for ShareableArray<'a, T, N>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ShareableArray::OwnedHeap(v) => v.serialize(serializer),
            ShareableArray::OwnedInline(v) => v.serialize(serializer),
            ShareableArray::Borrowed(v) => v.serialize(serializer),
            ShareableArray::Shared(v) => v.serialize(serializer),
        }
    }
}

impl<'a, T, const N: usize> ShareableArray<'a, T, N> {
    fn as_slice(&self) -> &[T] {
        match self {
            ShareableArray::OwnedHeap(v) => v,
            ShareableArray::OwnedInline(v) => v,
            ShareableArray::Borrowed(v) => v,
            ShareableArray::Shared(v) => v,
        }
    }
}
impl<'a, T, const N: usize> From<Vec<T>> for ShareableArray<'a, T, N> {
    fn from(v: Vec<T>) -> Self {
        ShareableArray::OwnedHeap(v)
    }
}
impl<'a, T, const N: usize> From<&'a [T]> for ShareableArray<'a, T, N> {
    fn from(v: &'a [T]) -> Self {
        ShareableArray::Borrowed(v)
    }
}
impl<'a, T, const N: usize> From<std::sync::Arc<[T]>> for ShareableArray<'a, T, N> {
    fn from(v: std::sync::Arc<[T]>) -> Self {
        ShareableArray::Shared(v)
    }
}
impl<'a, T, const N: usize> AsRef<[T]> for ShareableArray<'a, T, N> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> std::ops::Deref for ShareableArray<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
// impl comparison operators
impl<'a, T> PartialEq for ShareableArray<'a, T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<'a, T> PartialOrd for ShareableArray<'a, T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}
impl<'a, T> Eq for ShareableArray<'a, T> where T: Eq {}
impl<'a, T> Ord for ShareableArray<'a, T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct BinInfo<L> {
    // some bins operate on a finite set of categories, such as a label, while others are described using a less-than-obvious function
    pub id: Option<L>,
    pub valid_pipelines: ShareableArray<'static, usize>,
    pub rewards: ShareableArray<'static, f64>,
    pub costs: ShareableArray<'static, f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct AlgInputs<B, F, S> {
    pub binning_function: B,
    pub forecast_function: F,
    pub send_function: S,
}

pub trait LabelGroupingBehavior<L> {
    fn group_buckets<I>(bins: I) -> (Vec<usize>, Vec<BinInfo<L>>)
    where
        I: IntoIterator<Item = BinInfo<L>>;
}

#[derive(Debug, Clone, Copy)]
pub struct KeepUnique; // 00
impl<L> LabelGroupingBehavior<L> for KeepUnique {
    fn group_buckets<I>(bins: I) -> (Vec<usize>, Vec<BinInfo<L>>)
    where
        I: IntoIterator<Item = BinInfo<L>>,
    {
        let bins = bins.into_iter().collect::<Vec<_>>();
        let mut indices = Vec::with_capacity(bins.len());
        for i in 0..bins.len() {
            indices.push(i);
        }
        (indices, bins)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UniqueRewardMeanCost; // 01
                                 // TODO: implement this

#[derive(Debug, Clone, Copy)]
pub struct MeanRewardUniqueCost; // 10
                                 // TODO: implement this

#[derive(Debug, Clone, Copy)]
pub struct MeanRewardMeanCost; // 11
                               // we're going to group everything with the same label, and then get the mean of the rewards and costs and make a new bin out of that
impl<L: LabelRequirements> LabelGroupingBehavior<L> for MeanRewardMeanCost {
    fn group_buckets<I>(bins: I) -> (Vec<usize>, Vec<BinInfo<L>>)
    where
        I: IntoIterator<Item = BinInfo<L>>,
    {
        let bins = bins.into_iter();
        // let mut bins = bins.into_iter().collect::<Vec<_>>();
        // let mut indices = Vec::with_capacity(bins.len());
        let mut indices: Vec<usize> = Vec::new();
        let mut grouped_bins: Vec<BinInfo<L>> = Vec::new();
        type FastMap<K, V> = rustc_hash::FxHashMap<K, V>;
        let (low, _hi) = bins.size_hint();
        let mut grouped_bins_map: FastMap<L, (usize, usize)> = FastMap::with_capacity_and_hasher(
            low.ilog2().clamp(2, 10) as usize,
            Default::default(),
        );
        for bin in bins {
            if let Some(id) = &bin.id {
                if let Some((index, current_count)) = grouped_bins_map.get_mut(id) {
                    indices.push(*index);
                    *current_count += 1;
                    let current_bin = &mut grouped_bins[*index];
                    // current_bin.rewards.as_mut()[0] += bin.rewards.as_slice()[0];
                    for i in 0..current_bin.rewards.len() {
                        current_bin.rewards.as_mut()[i] += bin.rewards.as_slice()[i];
                        current_bin.costs.as_mut()[i] += bin.costs.as_slice()[i];
                    }
                    // this bin is discarded because its contribution has been logged
                } else {
                    let id = id.clone();
                    let bin_index = grouped_bins.len();
                    indices.push(grouped_bins.len());
                    grouped_bins.push(bin);
                    grouped_bins_map.insert(id, (bin_index, 1));
                }
            // None is not counted as equivalent to any bin, even itself
            } else {
                indices.push(grouped_bins.len());
                grouped_bins.push(bin);
            }
        }
        // we have our counts and sums, so now we make our means
        for bin in grouped_bins.iter_mut() {
            if let Some(id) = &bin.id {
                if let Some((_, count @ 1..)) = grouped_bins_map.get_mut(id) {
                    for i in 0..bin.rewards.len() {
                        bin.rewards.as_mut()[i] /= *count as f64;
                        bin.costs.as_mut()[i] /= *count as f64;
                    }
                }
                // Nones are unique and so their mean is the same as their sum
            }
        }
        (indices, grouped_bins)
    }
}

// tests for mean reward mean cost
#[cfg(test)]
mod test_mean_reward_mean_cost {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct TestLabel {
        id: usize,
        name: String,
    }

    #[test]
    fn test_group_buckets() {
        let bins = vec![
            BinInfo {
                id: Some(TestLabel {
                    id: 1,
                    name: "A".to_string(),
                }),
                valid_pipelines: ShareableArray::OwnedHeap(vec![0]),
                rewards: ShareableArray::OwnedHeap(vec![1.0]),
                costs: ShareableArray::OwnedHeap(vec![2.0]),
            },
            BinInfo {
                id: Some(TestLabel {
                    id: 1,
                    name: "A".to_string(),
                }),
                valid_pipelines: ShareableArray::OwnedHeap(vec![0]),
                rewards: ShareableArray::OwnedHeap(vec![2.0]),
                costs: ShareableArray::OwnedHeap(vec![3.0]),
            },
            BinInfo {
                id: Some(TestLabel {
                    id: 2,
                    name: "B".to_string(),
                }),
                valid_pipelines: ShareableArray::OwnedHeap(vec![2]),
                rewards: ShareableArray::OwnedHeap(vec![3.0]),
                costs: ShareableArray::OwnedHeap(vec![4.0]),
            },
        ];
        let (indices, grouped_bins) = MeanRewardMeanCost::group_buckets(bins);
        assert_eq!(indices, vec![0, 0, 1]);
        assert_eq!(grouped_bins.len(), 2);
        assert_eq!(grouped_bins[0].rewards.as_slice(), &[1.5]);
        assert_eq!(grouped_bins[0].costs.as_slice(), &[2.5]);
        assert_eq!(grouped_bins[1].rewards.as_slice(), &[3.0]);
        assert_eq!(grouped_bins[1].costs.as_slice(), &[4.0]);
    }
}

// WeightedMean,

pub trait LabelRequirements<G = MeanRewardMeanCost>
where
    Self: Debug + Clone + PartialEq + Eq + Hash + Serialize + DeserializeOwned,
    G: LabelGroupingBehavior<Self>,
{
}
impl<T> LabelRequirements for T where
    T: Debug + Clone + PartialEq + Eq + Hash + Serialize + DeserializeOwned
{
}

pub fn aquifer_scheduler<History, Label, BinFunc, ForecastFunc, SendFunc, SendError>(
    inputs: Vec<Tuple>,
    output_pipelines: &[AsyncPipe],
    history: &mut History,
    alg_inputs: AlgInputs<BinFunc, ForecastFunc, SendFunc>,
    budget: f64,
    future_kind: FutureWindowKind,
) -> Option<usize>
where
    Label: LabelRequirements,
    BinFunc: Fn(&Tuple) -> BinInfo<Label>,
    ForecastFunc:
        Fn(&mut History, &[BinInfo<Label>], FutureWindowKind) -> (Vec<BinInfo<Label>>, f64),
    SendFunc: Fn(
        &mut History,
        Vec<Tuple>,
        Vec<BinInfo<Label>>,
        usize,
        &[AsyncPipe],
    ) -> Result<(), SendError>,
    SendError: Debug,
{
    // use greedy
    lookahead_problem_scheduler(
        inputs,
        output_pipelines,
        history,
        alg_inputs,
        budget,
        future_kind,
        Strategy::Greedy,
    )
}
pub fn optimal_scheduler<History, Label, BinFunc, ForecastFunc, SendFunc, SendError>(
    inputs: Vec<Tuple>,
    output_pipelines: &[AsyncPipe],
    history: &mut History,
    alg_inputs: AlgInputs<BinFunc, ForecastFunc, SendFunc>,
    budget: f64,
    future_kind: FutureWindowKind,
) -> Option<usize>
where
    Label: LabelRequirements,
    BinFunc: Fn(&Tuple) -> BinInfo<Label>,
    ForecastFunc:
        Fn(&mut History, &[BinInfo<Label>], FutureWindowKind) -> (Vec<BinInfo<Label>>, f64),
    SendFunc: Fn(
        &mut History,
        Vec<Tuple>,
        Vec<BinInfo<Label>>,
        usize,
        &[AsyncPipe],
    ) -> Result<(), SendError>,
    SendError: Debug,
{
    // use optimal
    lookahead_problem_scheduler(
        inputs,
        output_pipelines,
        history,
        alg_inputs,
        budget,
        future_kind,
        Strategy::Optimal,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Greedy,
    Optimal,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment<L> {
    pub pipeline: usize,
    pub bin_info: BinInfo<L>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SwapInfo<L> {
    pub item_index: usize,
    pub starting_pipeline: usize,
    pub ending_pipeline: usize,
    pub score_diff: f64,
    pub swap_cost: f64,
    pub reward_diff: f64,
    pub bin_info: Option<L>,
}

pub fn lookahead_problem_scheduler<History, Label, BinFunc, ForecastFunc, SendFunc, SendError>(
    inputs: Vec<Tuple>,
    output_pipelines: &[AsyncPipe],
    history: &mut History,
    alg_inputs: AlgInputs<BinFunc, ForecastFunc, SendFunc>,
    budget: f64,
    future_kind: FutureWindowKind,
    strategy: Strategy,
) -> Option<usize>
where
    Label: LabelRequirements,
    BinFunc: Fn(&Tuple) -> BinInfo<Label>,
    ForecastFunc:
        Fn(&mut History, &[BinInfo<Label>], FutureWindowKind) -> (Vec<BinInfo<Label>>, f64),
    SendFunc: Fn(
        &mut History,
        Vec<Tuple>,
        Vec<BinInfo<Label>>,
        usize,
        &[AsyncPipe],
    ) -> Result<(), SendError>,
    SendError: Debug,
{
    // just call the other one with the default of MeanRewardMeanCost
    lookahead_problem_scheduler_with_bucket_grouping::<
        History,
        Label,
        MeanRewardMeanCost,
        BinFunc,
        ForecastFunc,
        SendFunc,
        SendError,
    >(
        inputs,
        output_pipelines,
        history,
        alg_inputs,
        budget,
        future_kind,
        strategy,
    )
}

use std::sync::atomic::{AtomicBool, AtomicUsize};
pub const EXPONENTIAL_WINDOW_CAP: usize = 10;
pub const DEBUG_USE_BOTH: bool = false;
pub const DEBUG_LOG_INPUTS: bool = true;

// controls whether greedy is available at all
// pub static USE_NEW_GREEDY: AtomicBool = AtomicBool::new(false);
pub static USE_NEW_GREEDY: AtomicBool = AtomicBool::new(true);
pub static DEDUPLICATE_BINS_THRESHOLD: AtomicUsize = AtomicUsize::new(100);

pub fn lookahead_problem_scheduler_with_bucket_grouping<
    History,
    Label,
    GroupingStrategy,
    BinFunc,
    ForecastFunc,
    SendFunc,
    SendError,
>(
    mut inputs: Vec<Tuple>,
    output_pipelines: &[AsyncPipe],
    history: &mut History,
    alg_inputs: AlgInputs<BinFunc, ForecastFunc, SendFunc>,
    mut budget: f64,
    mut future_kind: FutureWindowKind,
    strategy: Strategy,
) -> Option<usize>
where
    Label: LabelRequirements<GroupingStrategy>,
    GroupingStrategy: LabelGroupingBehavior<Label>,
    BinFunc: Fn(&Tuple) -> BinInfo<Label>,
    ForecastFunc:
        Fn(&mut History, &[BinInfo<Label>], FutureWindowKind) -> (Vec<BinInfo<Label>>, f64),
    SendFunc: Fn(
        &mut History,
        Vec<Tuple>,
        Vec<BinInfo<Label>>,
        usize,
        &[AsyncPipe],
    ) -> Result<(), SendError>,
    SendError: Debug,
{
    if let (Strategy::Optimal, FutureWindowKind::TimeMillis(time_millis)) = (strategy, future_kind)
    {
        // if we are using the optimal strategy, we need to warn if the future window doesn't have a fixed cap, otherwise it will be too slow
        warn!("optimal strategy was chosen with only a time window. Adding a fixed cap of {EXPONENTIAL_WINDOW_CAP} items into the future window");
        future_kind = FutureWindowKind::TimeWithMaximumCount {
            time_ms: time_millis,
            max_count: EXPONENTIAL_WINDOW_CAP,
        };
    }
    let future_kind = future_kind;

    // print parameters
    // let input_serialize = crate::SerdeJson(&inputs);
    let input_serialize = inputs.len();
    let output_pipeline_serialize = output_pipelines.len();
    info!(
        "aquifer_scheduler START \n\tinputs: {input_serialize:?}, \n\toutput_pipelines: <{output_pipeline_serialize:?} pipes>, \n\tinitial_budget: {budget}, \n\tfuture_kind: {future_kind:?}"
    );
    trace!("aquifer_scheduler g0");
    let bins: Vec<BinInfo<_>> = inputs
        .iter()
        .map(|t| (alg_inputs.binning_function)(t))
        .collect();
    trace!("aquifer_scheduler g1");
    let (forecast, ratio_kept): (Vec<BinInfo<_>>, f64) =
        (alg_inputs.forecast_function)(history, &bins, future_kind);
    trace!("aquifer_scheduler g2");
    if ratio_kept < 1.0 {
        let old_budget = budget;
        budget *= ratio_kept;
        info!(
            "Budget scaling: ratio_kept={:.4} (system pressure), old_budget={:.2}, new_budget={:.2}. Plan to drop {:.2}% of future load.",
            ratio_kept, old_budget, budget, 100.0 * (1.0 - ratio_kept)
        );
    } else {
        info!(
            "Budget scaling: ratio_kept is 1.0. System under capacity. Budget remains: {:.2}",
            budget
        );
    }

    // if the window is too small, the deduplicating greedy algorithm is not worth it
    let use_new_greedy = USE_NEW_GREEDY.load(std::sync::atomic::Ordering::Relaxed)
        && forecast.len() > DEDUPLICATE_BINS_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed);

    if strategy == Strategy::Greedy {
        info!(
            "Strategy selection: use_new_greedy={} (forecast_len={}, threshold={})",
            use_new_greedy,
            forecast.len(),
            DEDUPLICATE_BINS_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    let outputs: BTreeMap<usize, (Vec<Tuple>, Vec<BinInfo<Label>>)> = if use_new_greedy
        && strategy == Strategy::Greedy
    {
        let forecast_amount = forecast.len();
        // we reduce the buckets down so we have fewer swaps to consider
        debug!(
            "{} buckets before grouping: {:?}",
            forecast.len(),
            crate::SerdeJson(&forecast)
        );
        let (reduced_bucket_indices, reduced_buckets) = GroupingStrategy::group_buckets(forecast);
        debug!(
            "{} buckets after grouping: {:?}",
            reduced_buckets.len(),
            crate::SerdeJson(&reduced_buckets)
        );

        let mut swaps = make_redux_swaps(output_pipelines, &reduced_buckets);
        let mut assignments: Vec<Assignment<Label>> = (0..forecast_amount)
            .map(|original_index| Assignment {
                pipeline: 0,
                bin_info: reduced_buckets[reduced_bucket_indices[original_index]].clone(),
            })
            .collect();

        let greedy_start = std::time::Instant::now();
        let swaps_made = greedy_assign_redux(
            budget,
            &reduced_bucket_indices,
            &reduced_buckets,
            &mut swaps,
            &mut assignments,
        );
        let greedy_elapsed_micros = greedy_start.elapsed().as_nanos() as f64 / 1_000.0;

        // Audit Greedy Redux
        let mut total_cost = 0.0;
        let mut total_reward = 0.0;
        for a in &assignments {
            let bin = &a.bin_info;
            let pipeline_index = bin
                .valid_pipelines
                .iter()
                .position(|&p| p == a.pipeline)
                .unwrap_or(0);
            total_cost += bin.costs.get(pipeline_index).cloned().unwrap_or(0.0);
            total_reward += bin.rewards.get(pipeline_index).cloned().unwrap_or(0.0);
        }
        info!(
            "greedy_scheduler (redux): swaps={swaps_made}, time={:.3}ms, budget={:.2}, total_cost={:.2}, total_reward={:.2}, over_budget={}",
            greedy_elapsed_micros / 1000.0, budget, total_cost, total_reward, total_cost > budget
        );

        debug!(
            "greedy redux made {swaps_made} swaps in {:.5} ms",
            greedy_elapsed_micros / 1000.0
        );
        log_greedy_data(&inputs, swaps_made as _, greedy_elapsed_micros);

        let mut outputs: BTreeMap<usize, (Vec<Tuple>, Vec<BinInfo<Label>>)> = BTreeMap::new();
        let mut pipeline_counts = std::collections::HashMap::new();
        for (a, t) in assignments.into_iter().zip(inputs) {
            let pipeline_index = a.pipeline;
            *pipeline_counts.entry(pipeline_index).or_insert(0) += 1;
            let bin = a.bin_info;
            let (pipeline_outputs, pipeline_bins) = outputs
                .entry(pipeline_index)
                .or_insert_with(|| (Vec::new(), Vec::new()));
            pipeline_outputs.push(t);
            pipeline_bins.push(bin);
        }
        info!("Assignment summary (Greedy Redux): {:?}", pipeline_counts);
        outputs
    } else {
        let backup_forecast = if DEBUG_LOG_INPUTS {
            forecast.clone()
        } else {
            Vec::new()
        };

        let mut swaps =
            Vec::<SwapInfo<Label>>::with_capacity(output_pipelines.len() * forecast.len());
        for (item_index, bin) in forecast.iter().enumerate() {
            let BinInfo {
                valid_pipelines,
                rewards,
                costs,
                id: _,
            } = bin;
            let mut reward_check: Vec<(usize, f64, f64)> = valid_pipelines
                .iter()
                .zip(rewards.iter())
                .zip(costs.iter())
                .map(|((&pipeline_id, &reward), &cost)| (pipeline_id, reward, cost))
                .collect();
            reward_check.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // get indices where the cost is less than the previous one
            fn is_ascending<I: IntoIterator<Item = T>, T: Clone + PartialOrd>(i: I) -> bool {
                i.into_iter().tuple_windows().all(|(w0, w1)| w0 <= w1)
            }
            while !is_ascending(reward_check.iter().map(|v| v.2)) {
                // remove the first non-ascending element
                for i in 0..reward_check.len() - 1 {
                    if reward_check[i].2 > reward_check[i + 1].2 {
                        reward_check.remove(i);
                        break;
                    }
                }
            }
            // now we have a fully sorted, always beneficial order of pipelines for this bin
            // we can now calculate the swaps
            for window in reward_check.windows(2) {
                let [(starting_pipeline, starting_reward, starting_cost), (ending_pipeline, ending_reward, ending_cost)] =
                    window
                else {
                    panic!("window of scores was not of length 2");
                };
                let starting_score = if *starting_cost > 0.0 {
                    starting_reward / starting_cost
                } else {
                    if *starting_cost == 0.0 {
                        *starting_reward
                    } else {
                        warn!("starting cost was negative");
                        starting_reward / (-*starting_cost)
                    }
                };
                let ending_score = if *ending_cost > 0.0 {
                    ending_reward / ending_cost
                } else {
                    if *ending_cost == 0.0 {
                        *ending_reward
                    } else {
                        warn!("ending cost was negative");
                        ending_reward / (-*ending_cost)
                    }
                };
                let score_diff = ending_score - starting_score;
                let swap_cost = ending_cost - starting_cost;
                let reward_diff = ending_reward - starting_reward;
                if reward_diff < 0.0 {
                    error!("reward_diff was negative for bin {bin:?}");
                }
                swaps.push(SwapInfo {
                    item_index: item_index.clone(),
                    starting_pipeline: *starting_pipeline,
                    ending_pipeline: *ending_pipeline,
                    score_diff,
                    swap_cost,
                    reward_diff,
                    bin_info: bin.id.clone(),
                });
            }
        }
        trace!("aquifer_scheduler g3");

        // NaN was occurring above and propagating here,
        swaps.sort_by(|a, b| {
            a.score_diff.partial_cmp(&b.score_diff).unwrap_or_else(|| {
                if a.score_diff.is_nan() {
                    error!("score_diff {a:?} was NaN");
                    std::cmp::Ordering::Less
                } else {
                    error!("score_diff {b:?} was NaN");
                    std::cmp::Ordering::Greater
                }
            })
        });

        // TODO: this can be made more efficient by keeping only the label or the index to the bucket and then doing a lookup to a different argument
        // It requires changing how the bins are stored before being passed to the scheduler
        let mut assignments: Vec<Assignment<_>> = forecast
            .into_iter()
            .map(|t| Assignment {
                pipeline: 0,
                bin_info: t,
            })
            .collect();
        trace!("aquifer_scheduler g4");
        // debug!("before scheduling");

        if DEBUG_USE_BOTH {
            let mut swaps_clone = swaps.clone();
            let mut assignments1 = assignments.clone();
            let mut assignments2 = assignments.clone();
            let swaps_made = greedy_assign(budget, &mut swaps_clone, &mut assignments1);
            trace!("{swaps_made} swaps made in greedy algorithm");
            let optimal_table = optimal_assign_big_alloc(budget, &mut assignments2);

            let mut greedy_reward = 0.0;
            let mut greedy_cost = 0.0;
            for a in assignments1.iter() {
                let bin = &a.bin_info;
                let pipeline = a.pipeline;
                let Some(pipeline_index) = bin.valid_pipelines.iter().position(|v| *v == pipeline)
                else {
                    error!(
                        "greedy alg assigned invalid pipeline for an item in bin {:?}",
                        &bin.id
                    );
                    continue;
                };
                greedy_reward += bin.rewards[pipeline_index];
                greedy_cost += bin.costs[pipeline_index];
            }

            let mut optimal_reward = 0.0;
            let mut optimal_cost = 0.0;
            for a in assignments2.iter() {
                let bin = &a.bin_info;
                let pipeline = a.pipeline;
                let Some(pipeline_index) = bin.valid_pipelines.iter().position(|v| *v == pipeline)
                else {
                    error!(
                        "optimal alg assigned invalid pipeline for an item in bin {:?}",
                        &bin.id
                    );
                    continue;
                };
                optimal_reward += bin.rewards[pipeline_index];
                optimal_cost += bin.costs[pipeline_index];
            }

            // print everything
            let future_window_size = assignments.len();
            debug!("budget: {budget}, window size: {future_window_size}, greedy cost: {greedy_cost}, greedy reward: {greedy_reward}, optimal cost: {optimal_cost}, optimal reward: {optimal_reward}");

            let greedy_cost_bad = greedy_cost > budget;
            let optimal_cost_bad = optimal_cost > budget;
            let optimal_reward_bad = optimal_reward < greedy_reward;
            if greedy_cost_bad || optimal_cost_bad || optimal_reward_bad {
                let mut error_code = String::new();
                if greedy_cost_bad {
                    error_code.push_str("1");
                } else {
                    error_code.push_str("0");
                }
                if optimal_cost_bad {
                    error_code.push_str("1");
                } else {
                    error_code.push_str("0");
                }
                if optimal_reward_bad {
                    error_code.push_str("1");
                } else {
                    error_code.push_str("0");
                }
                // now we have to print out all the terrible debugging information
                let swaps_string = serde_json::to_string(&swaps).unwrap_or_else(|e| {
                    error!("failed to serialize swaps: {e}");
                    "".to_string()
                });
                let original_assignment_string = serde_json::to_string(&assignments)
                    .unwrap_or_else(|e| {
                        error!("failed to serialize original assignments: {e}");
                        "".to_string()
                    });
                let greedy_assignment_string =
                    serde_json::to_string(&assignments1).unwrap_or_else(|e| {
                        error!("failed to serialize greedy assignments: {e}");
                        "".to_string()
                    });
                let optimal_assignment_string = serde_json::to_string(&assignments2)
                    .unwrap_or_else(|e| {
                        error!("failed to serialize optimal assignments: {e}");
                        "".to_string()
                    });

                let optimal_table_string =
                    serde_json::to_string(&optimal_table).unwrap_or_else(|e| {
                        error!("failed to serialize optimal table: {e}");
                        "".to_string()
                    });
                error!("Scheduling error with code {error_code}\n\nswaps: {swaps_string}\n\noriginal assignments: {original_assignment_string}\n\ngreedy assignments: {greedy_assignment_string}\n\noptimal assignments: {optimal_assignment_string}\n\noptimal table: {optimal_table_string}\n");
            }
            if greedy_reward > optimal_reward {
                error!("greedy reward was greater than optimal reward");
                assignments = assignments1;
            } else {
                assignments = assignments2;
            }
        } else {
            match strategy {
                Strategy::Greedy => {
                    let greedy_start = std::time::Instant::now();
                    let swaps_made = greedy_assign(budget, &mut swaps, &mut assignments) as u64;
                    let greedy_elapsed_micros = greedy_start.elapsed().as_nanos() as f64 / 1_000.0;

                    // Audit Old Greedy
                    let mut total_cost = 0.0;
                    let mut total_reward = 0.0;
                    for a in &assignments {
                        let bin = &a.bin_info;
                        let pipeline_index = bin
                            .valid_pipelines
                            .iter()
                            .position(|&p| p == a.pipeline)
                            .unwrap_or(0);
                        total_cost += bin.costs.get(pipeline_index).cloned().unwrap_or(0.0);
                        total_reward += bin.rewards.get(pipeline_index).cloned().unwrap_or(0.0);
                    }
                    info!(
                        "greedy_scheduler (old): swaps={swaps_made}, time={:.3}ms, budget={:.2}, total_cost={:.2}, total_reward={:.2}, over_budget={}",
                        greedy_elapsed_micros / 1000.0, budget, total_cost, total_reward, total_cost > budget
                    );

                    debug!(
                        "old greedy made {swaps_made} swaps in {:.5} ms",
                        greedy_elapsed_micros / 1000.0
                    );
                    log_greedy_data(&inputs, swaps_made, greedy_elapsed_micros);
                }
                Strategy::Optimal => {
                    match optimal_assign_big_alloc(budget, &mut assignments) {
                        Ok(table) => {
                            trace!("aquifer_scheduler <optimal> g5");
                            trace!("table: {table:?}");

                            // Audit Optimal
                            let mut total_cost = 0.0;
                            let mut total_reward = 0.0;
                            for a in &assignments {
                                let bin = &a.bin_info;
                                let pipeline_index = bin
                                    .valid_pipelines
                                    .iter()
                                    .position(|&p| p == a.pipeline)
                                    .unwrap_or(0);
                                total_cost += bin.costs.get(pipeline_index).cloned().unwrap_or(0.0);
                                total_reward +=
                                    bin.rewards.get(pipeline_index).cloned().unwrap_or(0.0);
                            }
                            info!(
                                "optimal_scheduler: budget={:.2}, total_cost={:.2}, total_reward={:.2}, over_budget={}",
                                budget, total_cost, total_reward, total_cost > budget
                            );
                        }
                        Err(e) => {
                            error!("Failed to run optimal_assign_big_alloc when using original budget={budget}. Resetting assingments to minimal values after receiving error:\n{e:#?}");
                            if DEBUG_LOG_INPUTS {
                                debug_log_optimal_error(backup_forecast, e);
                            }
                            reset_to_minimal(&mut assignments);
                        }
                    }
                    trace!("aquifer_scheduler <optimal> g5");
                }
            }
        }

        trace!("final assignments: {:?}", assignments);
        let output_indices = assignments.iter().map(|a| a.pipeline).collect::<Vec<_>>();
        debug!("after scheduling. output_indices: {:?}", output_indices);

        let mut outputs: BTreeMap<usize, (Vec<Tuple>, Vec<BinInfo<Label>>)> = BTreeMap::new();
        let mut assignment_allowances: HashMap<Option<Label>, (usize, Vec<usize>)> = HashMap::new();
        let num_pipelines = output_pipelines.len();
        for (input_index, a) in assignments.iter().enumerate() {
            let (bin_allowance_total, individual_bin_allowances) = assignment_allowances
                .entry(a.bin_info.id.clone())
                .or_insert_with(|| (0, vec![0; num_pipelines]));
            let my_output = a.pipeline;
            let Some(pipeline_ref) = individual_bin_allowances.get_mut(my_output) else {
                error!("invalid pipeline index {my_output} (out of range for {num_pipelines} pipes) for input #{input_index} (which has the following bucket info {:?} )", a.bin_info);
                continue;
            };
            *pipeline_ref += 1;
            *bin_allowance_total += 1;
        }
        let input_pairs = inputs.drain(..).into_iter().zip(bins.into_iter());
        let mut rng = rand::thread_rng();
        for (input, input_bin) in input_pairs {
            use rand::Rng;
            let tuple_id = input.id();
            let output_pipeline = if let Some((total_allowance, per_pipeline_allowances)) =
                assignment_allowances.get_mut(&input_bin.id)
            {
                let allowances_to_skip = rng.gen_range(0..*total_allowance);
                let mut running_total = 0;
                let original_total = *total_allowance;
                let mut output = None;
                for (pipeline_index, allowance) in per_pipeline_allowances.iter_mut().enumerate() {
                    running_total += *allowance;
                    if running_total > allowances_to_skip {
                        *allowance -= 1;
                        *total_allowance -= 1;
                        output = Some(pipeline_index);
                        break;
                    }
                }
                if let Some(o) = output {
                    o
                } else {
                    warn!("no allowances were found when making random input for input tuple {tuple_id:?} (which has the following bucket info {input_bin:?} ). Generated index was {allowances_to_skip} but only {original_total} total allowances were available . dropping now");
                    // add it to the drop
                    0
                }
            } else {
                error!("no allowance entry was found for input {input:?}, dropping now");
                // add it to the drop
                0
            };

            let (tuple_vec, bin_vec) = outputs
                .entry(output_pipeline)
                .or_insert_with(|| (get_tuple_vec(), vec![]));
            tuple_vec.push(input);
            bin_vec.push(input_bin);
            continue;
        }

        // let mut outputs: BTreeMap<usize, (Vec<Tuple>, Vec<BinInfo<Label>>)> = BTreeMap::new();
        // let input_pairs = inputs.drain(..).into_iter().zip(bins.into_iter());
        // // input_pairs.sort_by_key // sort by the cost available in the item so it doesn't choose something that is above its limit
        // let mut rng = rand::thread_rng();
        // let mut applicable_assignments = Vec::new();
        // for (input, input_bin) in input_pairs {
        //     applicable_assignments.extend(assignments.iter().enumerate().filter_map(|(i, a)| {
        //         if &a.bin_info.id == &input_bin.id {
        //             Some((i, a.pipeline))
        //         } else {
        //             None
        //         }
        //     }));
        //     // applicable_assignments.retain // filter out any that are not available to the current item

        //     // drop if none are applicable
        //     if applicable_assignments.is_empty() {
        //         warn!("no applicable assignments were found for input {input:?}, dropping now");
        //         // add it to the drop
        //         let output_pipeline = 0;
        //         let (tuple_vec, bin_vec) = outputs
        //             .entry(output_pipeline)
        //             .or_insert_with(|| (get_tuple_vec(), vec![]));
        //         tuple_vec.push(input);
        //         bin_vec.push(input_bin);
        //         continue;
        //     }
        //     use rand::seq::SliceRandom;
        //     // select a random index from the applicable_assignments to be the match
        //     let output_pipeline = match applicable_assignments.choose(&mut rng) {
        //         Some((idx, pipeline)) => {
        //             //TODO: BUG: this removes the first time, but the length for the list is wrong after that. We either need a different data structure or we need to leave a hole in the list and a way to track that a certain value is invalid
        //             // either run the scheduler one at a time, or find a way to make it work properly
        //             assignments.swap_remove(*idx);
        //             *pipeline
        //         }
        //         None => {
        //             warn!("no applicable assignments were found for input {input:?}, dropping now");
        //             // add it to the drop
        //             0
        //         }
        //     };
        //     let (tuple_vec, bin_vec) = outputs
        //         .entry(output_pipeline)
        //         .or_insert_with(|| (get_tuple_vec(), vec![]));
        //     tuple_vec.push(input);
        //     bin_vec.push(input_bin);
        // }
        trace!("aquifer_scheduler g6");
        trace!("outputs: {:#?}", outputs);
        return_tuple_vec(inputs);
        outputs
    };
    let mut total_sent = 0;
    for (target_pipeline, (tuples, bins)) in outputs.into_iter() {
        let amount_to_send = tuples.len();
        info!(
            "sending {} tuples to pipeline {}",
            amount_to_send, target_pipeline
        );
        if let Err(e) =
            (alg_inputs.send_function)(history, tuples, bins, target_pipeline, output_pipelines)
        {
            error!(
                "failed to send {} tuples to pipeline {}: {:?}",
                amount_to_send, target_pipeline, e
            );
        } else {
            total_sent += amount_to_send;
        }
    }
    Some(total_sent)
}

fn make_redux_swaps<P, Label, GroupingStrategy>(
    output_pipelines: &[P],
    reduced_buckets: &[BinInfo<Label>],
) -> Vec<SwapInfoRedux>
where
    Label: LabelRequirements<GroupingStrategy>,
    GroupingStrategy: LabelGroupingBehavior<Label>,
{
    let mut swaps =
        Vec::<SwapInfoRedux>::with_capacity(output_pipelines.len() * reduced_buckets.len());
    for (reduced_bucket_index, bucket) in reduced_buckets.iter().enumerate() {
        let mut pipeline1 = 0;
        let mut pipeline2 = 1;
        while pipeline1 < bucket.valid_pipelines.len() && pipeline2 < bucket.valid_pipelines.len() {
            let starting_pipeline = bucket.valid_pipelines[pipeline1];
            let starting_reward = bucket.rewards[pipeline1];
            let starting_cost = bucket.costs[pipeline1];
            let ending_pipeline = bucket.valid_pipelines[pipeline2];
            let ending_reward = bucket.rewards[pipeline2];
            let ending_cost = bucket.costs[pipeline2];
            if starting_cost > ending_cost {
                // skip the current one
                pipeline1 = pipeline2;
                pipeline2 += 1;
                continue;
            }
            if starting_reward > ending_reward {
                // skip the next one
                pipeline2 += 1;
                continue;
            }
            // we made a valid swap, skipping over any bad stuff in the middle
            let starting_score = if starting_cost > 0.0 {
                starting_reward / starting_cost
            } else {
                if starting_cost == 0.0 {
                    starting_reward
                } else {
                    warn!("starting cost was negative");
                    starting_reward / (-starting_cost)
                }
            };
            let ending_score = if ending_cost > 0.0 {
                ending_reward / ending_cost
            } else {
                if ending_cost == 0.0 {
                    ending_reward
                } else {
                    warn!("ending cost was negative");
                    ending_reward / (-ending_cost)
                }
            };
            let score_diff = ending_score - starting_score;
            let swap_cost = ending_cost - starting_cost;
            let reward_diff = ending_reward - starting_reward;
            if reward_diff < 0.0 {
                error!("reward_diff was negative for bin {bucket:?}");
            }
            swaps.push(SwapInfoRedux {
                starting_pipeline,
                ending_pipeline,
                score_diff,
                swap_cost,
                reward_diff,
                bin_info: reduced_bucket_index,
            });
            // skip over any bad stuff in the middle
            pipeline1 = pipeline2;
            pipeline2 += 1;
        }
        //     for (j, &target) in bucket.valid_pipelines.iter().enumerate().skip(1) {
        //         let i = j - 1;
        //         let starting_pipeline = bucket.valid_pipelines[i];
        //         let starting_reward = bucket.rewards[i];
        //         let starting_cost = bucket.costs[i];
        //         let ending_pipeline = target;
        //         let ending_reward = bucket.rewards[j];
        //         let ending_cost = bucket.costs[j];
        //         let starting_score = if starting_cost > 0.0 {
        //             starting_reward / starting_cost
        //         } else {
        //             if starting_cost == 0.0 {
        //                 starting_reward
        //             } else {
        //                 warn!("starting cost was negative");
        //                 starting_reward / (-starting_cost)
        //             }
        //         };
        //         let ending_score = if ending_cost > 0.0 {
        //             ending_reward / ending_cost
        //         } else {
        //             if ending_cost == 0.0 {
        //                 ending_reward
        //             } else {
        //                 warn!("ending cost was negative");
        //                 ending_reward / (-ending_cost)
        //             }
        //         };
        //         let score_diff = ending_score - starting_score;
        //         let swap_cost = ending_cost - starting_cost;
        //         let reward_diff = ending_reward - starting_reward;
        //         if reward_diff < 0.0 {
        //             error!("reward_diff was negative for bin {bucket:?}");
        //         }
        //         swaps.push(SwapInfoRedux {
        //             starting_pipeline,
        //             ending_pipeline,
        //             score_diff,
        //             swap_cost,
        //             reward_diff,
        //             bin_info: reduced_bucket_index,
        //         });
        //     }
    }

    swaps.sort_by(|a, b| {
        a.score_diff.partial_cmp(&b.score_diff).unwrap_or_else(|| {
            // NaN always goes to the beginning of the list
            // this means they will never be considered before anything else
            if a.score_diff.is_nan() {
                error!("score_diff {a:?} was NaN");
                std::cmp::Ordering::Less
            } else {
                error!("score_diff {b:?} was NaN");
                std::cmp::Ordering::Greater
            }
        })
    });

    swaps
}

fn log_greedy_data(inputs: &Vec<crate::BetterTuple>, swaps_made: u64, greedy_elapsed_micros: f64) {
    trace!("aquifer_scheduler <greedy> g5");
    trace!("total swaps made: {}", swaps_made);
    for t in inputs.iter() {
        let tuple_id = t.id();
        let aux_data = Some(HashMap::from([
            (
                "greedy_swaps_made".to_raw_key(),
                crate::global_logger::LimitedHabValue::UnsignedInteger(swaps_made),
            ),
            (
                "greedy_elapsed_micros".to_raw_key(),
                crate::global_logger::LimitedHabValue::Float(greedy_elapsed_micros),
            ),
        ]));
        if let Err(e) =
            crate::global_logger::log_data(tuple_id, "greedy_algorithm_call".to_raw_key(), aux_data)
        {
            error!("failed to log greedy swaps made: {e:?}");
        }
    }
}

fn debug_log_optimal_error<L: LabelRequirements>(
    backup_forecast: Vec<BinInfo<L>>,
    e: OptimalAssignmentAlgorithmError<L>,
) {
    match serde_json::to_string_pretty(&backup_forecast) {
        Ok(input_serialized) => error!("forecast:\n{input_serialized}"),
        Err(e) => {
            error!("failed to serialize inputs: {e}");
            // return;
        }
    }

    match serde_json::to_string_pretty(&e) {
        Ok(e_serialized) => error!("error:\n{e_serialized}"),
        Err(e) => {
            error!("failed to serialize error: {e}");
            // return;
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SwapInfoRedux {
    pub starting_pipeline: usize,
    pub ending_pipeline: usize,
    pub score_diff: f64,
    pub swap_cost: f64,
    pub reward_diff: f64,
    // bin_info: &'a BinInfo<L>,
    pub bin_info: usize,
}

pub fn greedy_assign_redux<Label>(
    budget: f64,
    bucket_lookup_indices: &[usize],
    buckets: &[BinInfo<Label>],
    swaps: &mut Vec<SwapInfoRedux>,
    assignments: &mut Vec<Assignment<Label>>,
) -> u32
where
    Label: LabelRequirements,
{
    debug!("calling greedy_assign_redux with budget={budget}, {} buckets, {} unique swaps, and {} assignments",
        buckets.len(),
        swaps.len(),
        assignments.len()
    );
    let mut budget_remaining = budget;
    let mut swaps_made = 0;
    let mut swap_has_been_made = true;
    type Bucket = usize;
    type ModelList = Vec<usize>;
    // type FastMap<K, V> = std::collections::HashMap<K, V>;
    type FastMap<K, V> = nohash_hasher::IntMap<K, V>;
    // type FastMap<K, V> = rustc_hash::FxHashMap<K, V>;
    type BucketTotal = usize;
    let mut bucket_to_model_multiset: FastMap<Bucket, (BucketTotal, ModelList)> =
        FastMap::with_capacity_and_hasher(swaps.len() / 2, Default::default());
    let Some(max_valid_pipelines) = buckets
        .iter()
        .filter_map(|b| b.valid_pipelines.iter().copied().max())
        .max()
    else {
        error!("max_valid_pipelines was None");
        return 0;
    };
    let last_valid_pipeline_index = max_valid_pipelines + 1;
    for b in bucket_lookup_indices {
        let (bucket_total, assignments) = bucket_to_model_multiset
            .entry(*b)
            .or_insert_with(|| (0, vec![0; last_valid_pipeline_index]));
        // let lowest_pipeline = buckets[*b].valid_pipelines[0];
        // everything starts at skip
        *bucket_total += 1;
        assignments[0] += 1;
    }
    while swap_has_been_made && budget_remaining > 0.0 {
        swap_has_been_made = false;
        for swap in swaps.iter().rev() {
            let cost_diff = swap.swap_cost;
            if cost_diff > budget_remaining {
                continue;
            }
            match bucket_to_model_multiset.get_mut(&swap.bin_info) {
                // bucket total does not change
                Some((_bt, counts)) => {
                    // apply by removing an old assignment
                    if counts[swap.starting_pipeline] > 0 {
                        counts[swap.starting_pipeline] -= 1;
                    } else {
                        continue;
                    }
                }
                None => continue,
            }
            // and now we need to add the new assignment
            match bucket_to_model_multiset.get_mut(&swap.bin_info) {
                Some((_bt, counts)) => {
                    counts[swap.ending_pipeline] += 1;
                }
                None => {
                    error!(
                        "bucket_to_model_multiset was None for bucket {}",
                        swap.bin_info
                    );
                    continue;
                }
            }
            budget_remaining -= cost_diff;
            swaps_made += 1;
            swap_has_been_made = true;
            // we need to loop back around to the top of the list
            // to see if a higher-priority swap has now been opened up
            break;
        }
    }

    // use rand::seq::SliceRandom;
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // now we can take our multiset and apply everything
    'shuffle_valid_assignments: for (assignment_index, (bucket_lookup_index, assignment)) in
        bucket_lookup_indices
            .iter()
            .zip(assignments.iter_mut())
            .enumerate()
    {
        let Some((bucket_total, pipeline_list)) =
            bucket_to_model_multiset.get_mut(bucket_lookup_index)
        else {
            error!("bucket_to_model_multiset for assignment at index {assignment_index} was None for bucket {bucket_lookup_index} with info {:?}", &assignment.bin_info);
            assignment.pipeline = 0;
            continue;
        };
        // // method 1: pick random pipelines until we get a nonzero one
        // loop {
        //     let pipeline_index = rng.gen_range(0..pipeline_list.len());
        //     let pipeline_count = pipeline_list.get_mut(pipeline_index);
        //     if let Some(pipeline_count) = pipeline_count {
        //         if *pipeline_count > 0 {
        //             assignment.pipeline = pipeline_index;
        //             *pipeline_count -= 1;
        //             break;
        //         }
        //     } else {
        //         error!("pipeline was None");
        //         assignment.pipeline = 0;
        //         continue 'shuffle_valid_assignments;
        //     }
        // }

        // method 2: use the current total to get a random number and decrement that pipeline
        let mut dist_to_assignment_idx = rng.gen_range(0..*bucket_total);
        *bucket_total = bucket_total.saturating_sub(1);
        for p in 0..pipeline_list.len() {
            let pipeline_count = pipeline_list.get_mut(p);
            if let Some(pipeline_count) = pipeline_count {
                if *pipeline_count > 0 {
                    if dist_to_assignment_idx < *pipeline_count {
                        assignment.pipeline = p;
                        *pipeline_count = pipeline_count.saturating_sub(1);
                        continue 'shuffle_valid_assignments;
                    } else {
                        dist_to_assignment_idx -= *pipeline_count;
                    }
                }
            } else {
                error!("pipeline was None");
                assignment.pipeline = 0;
                continue 'shuffle_valid_assignments;
            }
        }
    }
    swaps_made
}

#[cfg(test)]
mod greedy_redux_tests {
    use super::*;

    mod paper_test_utils {
        use super::*;
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum PaperTestBuckets {
            P1,
            P2,
        }
        pub const P1: BinInfo<PaperTestBuckets> = BinInfo {
            id: Some(PaperTestBuckets::P1),
            valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2]),
            rewards: ShareableArray::Borrowed(&[0.0, 14.0, 18.0]),
            costs: ShareableArray::Borrowed(&[0.0, 1.0, 2.0]),
        };
        pub const P2: BinInfo<PaperTestBuckets> = BinInfo {
            id: Some(PaperTestBuckets::P2),
            valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2]),
            rewards: ShareableArray::Borrowed(&[0.0, 12.0, 15.0]),
            costs: ShareableArray::Borrowed(&[0.0, 1.0, 2.0]),
        };
    }

    #[test]
    fn test_greedy_assign_redux1() {
        let budget = 2.0;
        let bucket_lookup_indices = vec![1, 1, 0];
        let buckets: &[BinInfo<paper_test_utils::PaperTestBuckets>] =
            &[paper_test_utils::P1, paper_test_utils::P2];
        let mut swaps = make_redux_swaps(&[AsyncPipe::Dummy, AsyncPipe::Dummy], buckets);
        let mut assignments = bucket_lookup_indices
            .iter()
            .map(|b| &buckets[*b])
            .map(|b| Assignment {
                pipeline: 0,
                bin_info: b.clone(),
            })
            .collect::<Vec<_>>();
        let swaps_made = greedy_assign_redux(
            budget,
            &bucket_lookup_indices,
            &buckets,
            &mut swaps,
            &mut assignments,
        );
        assert_eq!(swaps_made, 2);
        let mut p1_pipeline0_count = 0;
        let mut p2_pipeline0_count = 0;
        let mut p1_pipeline1_count = 0;
        let mut p2_pipeline1_count = 0;
        let mut p1_pipeline2_count = 0;
        let mut p2_pipeline2_count = 0;
        let mut total_reward = 0.0;
        for assignment in assignments.iter() {
            match assignment.bin_info.id {
                Some(paper_test_utils::PaperTestBuckets::P1) => match assignment.pipeline {
                    0 => p1_pipeline0_count += 1,
                    1 => p1_pipeline1_count += 1,
                    2 => p1_pipeline2_count += 1,
                    _ => panic!("invalid pipeline"),
                },
                Some(paper_test_utils::PaperTestBuckets::P2) => match assignment.pipeline {
                    0 => p2_pipeline0_count += 1,
                    1 => p2_pipeline1_count += 1,
                    2 => p2_pipeline2_count += 1,
                    _ => panic!("invalid pipeline"),
                },
                None => panic!("invalid bin"),
            }
            total_reward += assignment.bin_info.rewards[assignment.pipeline];
        }
        println!("P1_pipeline0_count: {p1_pipeline0_count}, P2_pipeline0_count: {p2_pipeline0_count}, P1_pipeline1_count: {p1_pipeline1_count}, P2_pipeline1_count: {p2_pipeline1_count}, P1_pipeline2_count: {p1_pipeline2_count}, P2_pipeline2_count: {p2_pipeline2_count}");
        // print all the assignments
        println!("{:#?}", assignments);
        assert_eq!(p1_pipeline0_count, 0);
        assert_eq!(p2_pipeline0_count, 1);
        assert_eq!(p1_pipeline1_count, 1);
        assert_eq!(p2_pipeline1_count, 1);
        assert_eq!(p1_pipeline2_count, 0);
        assert_eq!(p2_pipeline2_count, 0);
        assert_eq!(total_reward, 26.0);
    }

    #[test]
    fn test_greedy_assign_redux2() {
        let budget = 4.0;
        let bucket_lookup_indices = vec![1, 0, 0];
        let buckets: &[BinInfo<paper_test_utils::PaperTestBuckets>] =
            &[paper_test_utils::P1, paper_test_utils::P2];
        let mut swaps = make_redux_swaps(&[AsyncPipe::Dummy, AsyncPipe::Dummy], buckets);
        let mut assignments = bucket_lookup_indices
            .iter()
            .map(|b| &buckets[*b])
            .map(|b| Assignment {
                pipeline: 0,
                bin_info: b.clone(),
            })
            .collect::<Vec<_>>();
        let swaps_made = greedy_assign_redux(
            budget,
            &bucket_lookup_indices,
            &buckets,
            &mut swaps,
            &mut assignments,
        );
        assert_eq!(swaps_made, 4);
        let mut p1_pipeline0_count = 0;
        let mut p2_pipeline0_count = 0;
        let mut p1_pipeline1_count = 0;
        let mut p2_pipeline1_count = 0;
        let mut p1_pipeline2_count = 0;
        let mut p2_pipeline2_count = 0;
        let mut total_reward = 0.0;
        for assignment in assignments.iter() {
            match assignment.bin_info.id {
                Some(paper_test_utils::PaperTestBuckets::P1) => match assignment.pipeline {
                    0 => p1_pipeline0_count += 1,
                    1 => p1_pipeline1_count += 1,
                    2 => p1_pipeline2_count += 1,
                    _ => panic!("invalid pipeline"),
                },
                Some(paper_test_utils::PaperTestBuckets::P2) => match assignment.pipeline {
                    0 => p2_pipeline0_count += 1,
                    1 => p2_pipeline1_count += 1,
                    2 => p2_pipeline2_count += 1,
                    _ => panic!("invalid pipeline"),
                },
                None => panic!("invalid bin"),
            }
            total_reward += assignment.bin_info.rewards[assignment.pipeline];
        }
        println!("P1_pipeline0_count: {p1_pipeline0_count}, P2_pipeline0_count: {p2_pipeline0_count}, P1_pipeline1_count: {p1_pipeline1_count}, P2_pipeline1_count: {p2_pipeline1_count}, P1_pipeline2_count: {p1_pipeline2_count}, P2_pipeline2_count: {p2_pipeline2_count}");
        // print all the assignments
        println!("{:#?}", assignments);
        assert_eq!(p1_pipeline0_count, 0);
        assert_eq!(p2_pipeline0_count, 0);
        assert_eq!(p1_pipeline1_count, 2);
        assert_eq!(p2_pipeline1_count, 0);
        assert_eq!(p1_pipeline2_count, 0);
        assert_eq!(p2_pipeline2_count, 1);
        assert_eq!(total_reward, 43.0);
    }
}

pub fn greedy_assign<Label>(
    budget: f64,
    swaps: &mut Vec<SwapInfo<Label>>,
    assignments: &mut Vec<Assignment<Label>>,
) -> u32
where
    Label: LabelRequirements,
{
    let mut budget_remaining = budget;
    let mut swaps_made = 0;
    let mut swap_has_been_made = true;
    trace!("aquifer_scheduler g4.5 with\n\tassignments: {assignments:?}\n\tswaps: {swaps:?}\n\tbudget_remaining: {budget_remaining}");
    while swap_has_been_made && budget_remaining > 0.0 {
        swap_has_been_made = false;
        for swap in swaps.iter().rev() {
            let cost_diff = swap.swap_cost;
            if cost_diff > budget_remaining {
                continue;
            }
            if assignments[swap.item_index].pipeline != swap.starting_pipeline {
                continue;
            }
            trace!("performed swap: {:?}", swap);
            assignments[swap.item_index].pipeline = swap.ending_pipeline;
            budget_remaining -= cost_diff;
            swaps_made += 1;
            swap_has_been_made = true;
            break;
        }
    }
    swaps_made
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
enum OptimalAssignmentAlgorithmError<Label> {
    NoAssignments,
    NoValidPipelines,
    MaxValidPipelinesZero,
    NoMemoizationStart(OptimalMemTable),
    BestRewardNegative(OptimalMemTable),
    SlotwiseLocalRewardMismatch {
        mem_table: SlotwiseMemTable,
        assignment_index: usize,
        assignment_pipeline: usize,
        expected_reward: f64,
        actual_reward: f64,
    },
    SlotwiseLocalCostMismatch {
        mem_table: SlotwiseMemTable,
        assignment_index: usize,
        assignment_pipeline: usize,
        expected_cost: f64,
        actual_cost: f64,
    },
    BinwiseLocalRewardMismatch {
        mem_table: BinwiseMemTable,
        assignment_index: usize,
        assignment_pipeline: usize,
        expected_reward: f64,
        actual_reward: f64,
    },
    BinwiseLocalCostMismatch {
        mem_table: BinwiseMemTable,
        assignment_index: usize,
        assignment_pipeline: usize,
        expected_cost: f64,
        actual_cost: f64,
    },
    NoNextAssginment(OptimalMemTable),
    BudgetValidityError {
        mem_table: OptimalMemTable,
        budget_error: CheckBudgetError<Label>,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlotwiseMemTable {
    mem_table: ndarray::Array3<f64>,
    num_slots: usize,
    num_pipelines: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinwiseMemTable {
    mem_table: ndarray::Array4<f64>,
    num_slots: usize,
    num_bins: usize,
    num_pipelines: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
enum OptimalMemTable {
    Slotwise(SlotwiseMemTable),
    Binwise(BinwiseMemTable),
}

// check that the cost for assignments so far is less than the budget, and if it is greater than or equal to
// the budget, then we have a bug in the algorithm and need to reset the whole thing
#[derive(Debug, Clone, Serialize, Deserialize)]
enum CheckBudgetError<Label> {
    BudgetExceeded {
        limit: f64,
        actual: f64,
    },
    LengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidPipeline {
        index: usize,
        scheduled_pipeline: usize,
        bin_info: BinInfo<Label>,
    },
}

fn check_budget<Label: Debug + Clone>(
    budget: f64,
    assignments: &[Assignment<Label>],
    potential_pipelines: &[usize],
) -> Result<f64, CheckBudgetError<Label>> {
    if potential_pipelines.len() != assignments.len() {
        return Err(CheckBudgetError::LengthMismatch {
            expected: assignments.len(),
            actual: potential_pipelines.len(),
        });
    }
    let mut total_cost = 0.0;
    for (a_index, assignment) in assignments.iter().enumerate() {
        let pipeline = potential_pipelines[a_index];
        let bin_info = &assignment.bin_info;
        let Some(pipeline_index) = bin_info.valid_pipelines.iter().position(|v| *v == pipeline)
        else {
            return Err(CheckBudgetError::InvalidPipeline {
                index: a_index,
                scheduled_pipeline: pipeline,
                bin_info: bin_info.clone(),
            });
        };
        total_cost += bin_info.costs[pipeline_index];
    }
    if total_cost > budget {
        warn!("budget exceeded:\nassignments: {assignments:#?}\npotential_pipelines: {potential_pipelines:?}\nbudget: {budget}, total_cost: {total_cost}");
        return Err(CheckBudgetError::BudgetExceeded {
            limit: budget,
            actual: total_cost,
        });
    }
    Ok(total_cost)
}

// when we have received an error, we still need to schedule the items,
// let's schedule as many as we can to the minimum possible cost
fn reset_to_minimal<Label: Debug>(assignments: &mut Vec<Assignment<Label>>) {
    for assignment in assignments {
        let Some(&pipeline) = assignment.bin_info.valid_pipelines.first() else {
            error!("no valid pipelines for bin {:?}", &assignment.bin_info.id);
            assignment.pipeline = 0;
            continue;
        };
        assignment.pipeline = pipeline;
    }
}

// check if it's good, otherwise reset
fn check_and_reset<Label: Debug + Clone>(
    budget: f64,
    assignments: &mut Vec<Assignment<Label>>,
    potential_pipelines: &[usize],
) -> Result<(), CheckBudgetError<Label>> {
    match check_budget(budget, assignments, potential_pipelines) {
        Ok(_) => Ok(()),
        Err(e) => {
            error!("check budget error: {e:?}");
            reset_to_minimal(assignments);
            Err(e)
        }
    }
}

fn optimal_assign_big_alloc<Label>(
    budget: f64,
    assignments: &mut Vec<Assignment<Label>>,
) -> Result<OptimalMemTable, OptimalAssignmentAlgorithmError<Label>>
where
    Label: LabelRequirements,
{
    if assignments.is_empty() {
        error!("no assignments");
        return Err(OptimalAssignmentAlgorithmError::NoAssignments);
    }

    let Some(max_valid_pipelines) = assignments
        .iter()
        .map(|a| {
            (
                &a.bin_info,
                a.bin_info.valid_pipelines.iter().copied().max(),
            )
        })
        .inspect(|(bin, item)| {
            if item.is_none() {
                warn!("max valid pipelines was None because bin {bin:?} had no valid pipelines");
            }
        })
        .filter_map(|v| v.1.map(|v| v + 1))
        .max()
    else {
        error!("no assignments");
        return Err(OptimalAssignmentAlgorithmError::NoAssignments);
    };

    trace!("max valid pipelines: {max_valid_pipelines}");

    if max_valid_pipelines == 0 {
        error!("max valid pipelines was zero");
        return Err(OptimalAssignmentAlgorithmError::MaxValidPipelinesZero);
    }

    let avg_valid_pipelines = assignments
        .iter()
        .map(|a| a.bin_info.valid_pipelines.len() as f64)
        .sum::<f64>()
        / (assignments.len() as f64);

    let num_slots = assignments.len();
    // deduplicate and get our count
    let mut bins: HashMap<Option<Label>, (BinInfo<Label>, usize)> = Default::default();
    for a in assignments.iter() {
        let entry = bins
            .entry(a.bin_info.id.clone())
            .or_insert((a.bin_info.clone(), 0));
        entry.1 += 1;
    }
    let num_bins = bins.len();
    let mut new_bins = Vec::with_capacity(num_bins);
    let mut bin_counts = Vec::with_capacity(num_bins);
    let mut max_bin_count = 0;
    // let mut bin_lookup = HashMap::new();
    for (_bin_index, (bin_id, (bin, count))) in bins.into_iter().enumerate() {
        //     bin_lookup.insert(bin_id.clone(), bin_index);
        new_bins.push((bin_id, bin));
        bin_counts.push(count);
        max_bin_count = max_bin_count.max(count);
    }
    let bins = new_bins;

    trace!("optimal_assign_big_alloc g0");
    // // a 2d array where we get the cost and reward of each assignment
    // // it is at most the maximum number of valid pipelines for a bin by the number of bins
    // let mut shared_bin_costs = ndarray::Array2::<f64>::ones([num_bins, max_valid_pipelines]);
    // // negative reward will indicate that the scheduler should just not consider it at all
    // let mut shared_bin_rewards = -ndarray::Array2::<f64>::ones([num_bins, max_valid_pipelines]);
    // for bin_id in 0..num_bins {
    //     let (_, bin) = &bins[bin_id];
    //     let valid_pipelines = &bin.valid_pipelines;
    //     for (i, &pipeline_index) in valid_pipelines.iter().enumerate() {
    //         let cost = bin.costs[i];
    //         let reward = bin.rewards[i];
    //         trace!("optimal_assign_big_alloc g1half.loop.{bin_id}.loop.{pipeline_index}");
    //         shared_bin_costs[[bin_id, pipeline_index]] = cost;
    //         shared_bin_rewards[[bin_id, pipeline_index]] = reward;
    //     }
    // }
    trace!("optimal_assign_big_alloc g1");

    // Here bins is defined as b, number of models/pipelines is defined as p, and number of future slots to budget is defined as n.
    // We need to find the tradeoff between the number of bins, the number of pipelines, and the number of slots to decide which exp algorithm to use
    // One expansion is O(n^(bp)) and the other is O(p^n). Until there's concrete data, we ignore coefficients and just look at the exponential factors.
    // we will need O*2 space because we need to store the cost and reward of each assignment, in addition to each possible assignment
    let bin_considerations = (avg_valid_pipelines * (num_bins as f64)).ceil() as u32;
    // We can optimize to max_bin_count <= n because the worst case here is that one bin holds everything.
    // The best case is that we get to cut down on n by a factor of b, so we go from n^(bp) to (n/b)^(bp)
    let notation1_est_cost = max_bin_count.pow(bin_considerations);
    // p^n
    let notation2_est_cost = max_valid_pipelines.pow(num_slots as u32);
    let last_dim = 2;

    const USE_SLOTWISE_EXPSANSION: bool = true;
    const CLOSENESS_THRESHOLD: f64 = 0.05;
    const WARNING_THRESHOLD: f64 = 0.01;
    if notation1_est_cost < notation2_est_cost || USE_SLOTWISE_EXPSANSION {
        trace!("optimal_assign_big_alloc g2.a");
        // We have a row representing the current item being assigned
        // We have a column representing the model/pipeline to run it with
        // and in the third dimension, we have the data for the cell, which is the cost and reward of the best assignment so far
        let dims = [num_slots, max_valid_pipelines, last_dim];
        let mut mem_table: ndarray::Array3<f64> = ndarray::Array3::zeros(dims);
        let mut best_reward = 0.0;
        let mut best_cost = f64::INFINITY;
        let starting_index = 0;
        let mut best_start_index = None;
        // let's see if the budget violates our minimum cost
        let mut found_possible = false;
        'cost_is_possible: for bin_index in 0..num_bins {
            let bin = &bins[bin_index];
            let valid_pipelines = &bin.1.valid_pipelines;
            let costs = &bin.1.costs;
            for (&p, &c) in valid_pipelines.iter().zip(costs.iter()) {
                if p == 0 {
                    continue;
                }
                if c <= budget {
                    found_possible = true;
                    break 'cost_is_possible;
                }
            }
        }
        if !found_possible {
            warn!("budget of {} was too low to assign any values", budget);
            return Ok(OptimalMemTable::Slotwise(SlotwiseMemTable {
                mem_table,
                num_slots,
                num_pipelines: max_valid_pipelines,
            }));
        }
        // we have to loop to roll the first layer out here because we can't recurse into it without a place to start
        // for possible_assignment in 0..max_valid_pipelines {
        for possible_assignment in (0..max_valid_pipelines).rev() {
            trace!("optimal_assign_big_alloc g3.a.loop.{possible_assignment}.1");
            let mut local_best_reward = 0.0;
            let (cost, reward) = optimal_assign_pipeline_per_item(
                0.0,
                0.0,
                budget,
                &assignments,
                starting_index,
                possible_assignment,
                &mut mem_table,
                // &bin_lookup,
                // &shared_bin_costs,
                // &shared_bin_rewards,
                &dims,
                &mut local_best_reward,
            );
            trace!("optimal_assign_big_alloc g3.a.loop.{possible_assignment}.2");
            if reward < 0.0 {
                trace!("reward was negative, skipping this possible assignment");
                continue;
            }
            if reward >= best_reward {
                best_reward = reward;
                best_cost = cost;
                best_start_index = Some(possible_assignment);
                trace!("optimal_assign_big_alloc g3.a.loop.{possible_assignment}.3");
            }
            trace!("optimal_assign_big_alloc g3.a.loop.{possible_assignment}.4");
        }
        let Some(best_start_index) = best_start_index else {
            error!("no best start index was found");
            return Err(OptimalAssignmentAlgorithmError::NoMemoizationStart(
                OptimalMemTable::Slotwise(SlotwiseMemTable {
                    mem_table,
                    num_slots,
                    num_pipelines: max_valid_pipelines,
                }),
            ));
        };
        if best_reward < 0.0 {
            warn!("best reward was zero or negative: {best_reward}");
            return Err(OptimalAssignmentAlgorithmError::BestRewardNegative(
                OptimalMemTable::Slotwise(SlotwiseMemTable {
                    mem_table,
                    num_slots,
                    num_pipelines: max_valid_pipelines,
                }),
            ));
        }

        trace!("optimal_assign_big_alloc g4");

        // now we use the best start index to build back out the best assignment path by following the memoization table
        let mut best_assignments = Vec::with_capacity(num_slots);

        let reward_warning_threshold_amount = best_reward * WARNING_THRESHOLD;
        let cost_warning_threshold_amount = best_cost * WARNING_THRESHOLD;
        // recursively go through the memoization table to find the best path
        fn find_best_path<L: Hash + Eq>(
            mem_table: &ndarray::Array3<f64>,
            best_reward: f64,
            best_cost: f64,
            reward_threshold: f64,
            remaining_budget: f64,
            cost_threshold: f64,
            // shared_bin_rewards: &ndarray::Array2<f64>,
            // shared_bin_costs: &ndarray::Array2<f64>,
            assignments: &[Assignment<L>],
            // bin_lookup: &HashMap<Option<L>, usize>,
            path_so_far: &mut Vec<usize>,
        ) -> bool {
            trace!("path so far: {:?}", path_so_far);
            let current_item = path_so_far.len();
            match current_item.cmp(&mem_table.shape()[0]) {
                // we can keep going
                std::cmp::Ordering::Less => {}
                // we've reached the end
                std::cmp::Ordering::Equal => {
                    trace!("we're at the end of a path. remaining budget: {remaining_budget}, cost threshold: {cost_threshold}");
                    // return remaining_budget >= 0.0 || remaining_budget.abs() <= cost_threshold;
                    return remaining_budget.abs() <= cost_threshold;
                }
                // something has gone horribly wrong
                std::cmp::Ordering::Greater => panic!("out of bounds in find_best_path"),
            }

            // let current_bin = &assignments[current_item].bin_info;
            // let current_bin_index = bin_lookup[&current_bin.id];

            // for next_possible_pipeline in (0..mem_table.shape()[1]).rev() {
            for next_possible_pipeline in 0..mem_table.shape()[1] {
                // let cost = shared_bin_costs[[current_bin_index, next_possible_pipeline]];
                let cost = assignments[current_item].bin_info.costs[next_possible_pipeline];

                let projected_total_reward = mem_table[[current_item, next_possible_pipeline, 1]];
                let projected_total_cost = mem_table[[current_item, next_possible_pipeline, 0]];
                if projected_total_reward < 0.0 {
                    continue;
                }
                if (projected_total_reward - best_reward).abs() > reward_threshold {
                    continue;
                }
                trace!("reward was right for pipeline {next_possible_pipeline}");

                if (projected_total_cost - best_cost).abs() > cost_threshold {
                    trace!("cost was too high for pipeline {next_possible_pipeline} at {projected_total_cost}, skipping");
                    continue;
                }
                let next_budget = remaining_budget - cost;
                if next_budget <= (0.0 - cost_threshold) {
                    trace!("next budget was negative at {next_budget}, skipping");
                    continue;
                }
                path_so_far.push(next_possible_pipeline);
                if find_best_path(
                    mem_table,
                    best_reward,
                    best_cost,
                    reward_threshold,
                    next_budget,
                    cost_threshold,
                    // shared_bin_rewards,
                    // shared_bin_costs,
                    assignments,
                    // bin_lookup,
                    path_so_far,
                ) {
                    return true;
                }
                path_so_far.pop();
            }

            // if we found nothing then we're in a bad state. the parent will pop us off the stack
            false
        }
        let mut path_so_far = vec![];
        let found_path = find_best_path(
            &mem_table,
            best_reward,
            best_cost,
            reward_warning_threshold_amount,
            // budget,
            best_cost,
            cost_warning_threshold_amount,
            // &shared_bin_rewards,
            // &shared_bin_costs,
            &*assignments,
            // &bin_lookup,
            &mut path_so_far,
        );
        if !found_path {
            error!("no path was found");
            return Err(OptimalAssignmentAlgorithmError::NoNextAssginment(
                OptimalMemTable::Slotwise(SlotwiseMemTable {
                    mem_table,
                    num_slots,
                    num_pipelines: max_valid_pipelines,
                }),
            ));
        }
        best_assignments.extend(path_so_far);

        trace!("optimal_assign_big_alloc g5");
        // check that the assignments are valid
        if let Err(e) = check_and_reset(budget, assignments, &best_assignments) {
            error!("check and reset error: {e:?}");
            return Err(OptimalAssignmentAlgorithmError::NoMemoizationStart(
                OptimalMemTable::Slotwise(SlotwiseMemTable {
                    mem_table,
                    num_slots,
                    num_pipelines: max_valid_pipelines,
                }),
            ));
        }
        trace!("optimal_assign_big_alloc g6");
        // assign all of them now that we know they're good
        for (a, &p) in assignments.iter_mut().zip(best_assignments.iter()) {
            trace!(
                "optimal_assign_big_alloc g7.loop.{:?}, : {p}",
                &a.bin_info.id
            );
            a.pipeline = p;
        }
        trace!("optimal_assign_big_alloc g8");
        Ok(OptimalMemTable::Slotwise(SlotwiseMemTable {
            mem_table,
            num_slots,
            num_pipelines: max_valid_pipelines,
        }))
    } else {
        trace!("optimal_assign_big_alloc g2.b");
        // the alternative to slotwise expansion is bin-model-wise expansion
        // this should be more space efficient than slotwise expansion,
        // but it doesn't start getting to be more efficient until the number of slots is significantly greater than the number of bins

        // we store the cost and the reward of the best assignment for that path
        // in a column representing the model
        // and a row representing the current item being assigned
        let dims = [num_bins, max_valid_pipelines, max_bin_count + 1, last_dim];
        let mut mem_table: ndarray::Array4<f64> = -ndarray::Array4::ones(dims);

        // we have to loop to roll the first layer out here because we can't recurse into it without a place to start
        // so we loop through all possible allocations of the first one
        let start_bin_index = 0;
        let start_pipeline_index = 0;
        let bin_count = bin_counts[start_bin_index];

        // let bin_costs = shared_bin_costs[[start_bin_index, start_pipeline_index]];
        // let bin_rewards = shared_bin_rewards[[start_bin_index, start_pipeline_index]];

        let bin_costs = assignments[start_bin_index].bin_info.costs[start_pipeline_index];
        let bin_rewards = assignments[start_bin_index].bin_info.rewards[start_pipeline_index];

        let mut max_assignment_amount = bin_count;
        let amount_can_fit = (budget / bin_costs).floor() as usize;
        max_assignment_amount = max_assignment_amount.min(amount_can_fit);
        if bin_rewards < 0.0 {
            max_assignment_amount = 0;
        }
        if max_assignment_amount > max_bin_count {
            warn!(
                "max assignment amount was greater than max bin count, setting it to max bin count"
            );
            max_assignment_amount = max_bin_count;
        }

        trace!("optimal_assign_big_alloc g3.b");
        let mut best_reward = 0.0;
        let mut best_cost = 0.0;
        let mut best_start_index = None;
        let bin_index_lookup = bins
            .iter()
            .enumerate()
            .map(|(i, (b, _))| (b, i))
            .collect::<HashMap<_, _>>();
        let bins = bins.iter().map(|(_, b)| b.clone()).collect::<Vec<_>>();
        for possible_assignment_amount in 0..=max_assignment_amount {
            let mut local_best_reward = 0.0;
            let (cost, reward) = optimal_assign_count_per_pipeline_per_bin(
                0.0,
                0.0,
                budget,
                &bins,
                &bin_counts,
                start_bin_index,
                start_pipeline_index,
                possible_assignment_amount,
                0,
                0,
                num_slots,
                &mut mem_table,
                // &shared_bin_costs,
                // &shared_bin_rewards,
                &dims,
                &mut local_best_reward,
            );
            if reward < 0.0 {
                trace!("reward was negative, skipping this possible assignment");
                continue;
            }
            if reward >= best_reward {
                best_reward = reward;
                best_cost = cost;
                best_start_index = Some(possible_assignment_amount);
            }
        }
        trace!("optimal_assign_big_alloc g4.b");
        let Some(best_start_index) = best_start_index else {
            error!("no best start index was found");
            return Err(OptimalAssignmentAlgorithmError::NoMemoizationStart(
                OptimalMemTable::Binwise(BinwiseMemTable {
                    mem_table,
                    num_slots,
                    num_bins,
                    num_pipelines: max_valid_pipelines,
                }),
            ));
        };
        if best_reward < 0.0 {
            warn!("best reward was zero or negative: {best_reward}");
            return Err(OptimalAssignmentAlgorithmError::BestRewardNegative(
                OptimalMemTable::Binwise(BinwiseMemTable {
                    mem_table,
                    num_slots,
                    num_bins,
                    num_pipelines: max_valid_pipelines,
                }),
            ));
        }

        // now we use the best start index to build back out the best assignment path by following the memoization table
        let mut assignment_counts: ndarray::Array2<usize> =
            ndarray::Array2::zeros([num_bins, max_valid_pipelines]);
        let mut current_bin = start_bin_index;
        let mut current_pipeline = start_pipeline_index;
        let mut current_count = best_start_index;

        trace!("optimal_assign_big_alloc g5.b");

        'assign_next_count: loop {
            if current_bin >= num_bins {
                error!("current bin was out of bounds. stopping allocation");
                break;
            }
            if current_pipeline >= max_valid_pipelines {
                error!("current pipeline was out of bounds. stopping allocation");
                break;
            }
            if current_count > max_bin_count {
                error!("current count was greater than max bin count, stopping allocation");
                break;
            }
            trace!("optimal_assign_big_alloc g6.b.loop.1");

            assignment_counts[[current_bin, current_pipeline]] = current_count;

            current_pipeline += 1;
            if current_pipeline >= max_valid_pipelines {
                current_pipeline = 0;
                current_bin += 1;
                if current_bin >= num_bins {
                    break;
                }
            }
            trace!("optimal_assign_big_alloc g6.b.loop.2");

            let reward_warning_threshold_amount = best_reward * WARNING_THRESHOLD;
            let mut closest_match = None;
            let mut closest_diff = f64::INFINITY;
            for p in 0..=max_bin_count {
                trace!("optimal_assign_big_alloc g6.b.loop.3.{p}");
                let reward = mem_table[[current_bin, current_pipeline, p, 1]];
                let diff = (reward - best_reward).abs();
                if diff < closest_diff {
                    closest_diff = diff;
                    closest_match = Some(p);
                    // we found it, no need to iterate through the rest
                    if diff <= f64::EPSILON {
                        break;
                    }
                }
            }
            if let Some(p) = closest_match {
                if closest_diff <= reward_warning_threshold_amount {
                    current_count = p;
                    continue 'assign_next_count;
                }
            }
            trace!("optimal_assign_big_alloc g6.b.loop.4");

            warn!("no valid pipeline was found for bin {current_bin}, pipeline {current_pipeline}, count {current_count}");
            break;
        }

        let mut remaining_bin_counts = bin_counts.clone();
        let mut best_assignments = Vec::with_capacity(num_slots);
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for (a_index, a) in assignments.iter().enumerate() {
            trace!("optimal_assign_big_alloc g7.b.loop.1.{a_index}");
            let Some(&bin_index) = bin_index_lookup.get(&a.bin_info.id) else {
                error!(
                    "bin index was not found for bin {:?}, skipping",
                    &a.bin_info.id
                );
                continue;
            };
            let remaining_bin_count = remaining_bin_counts[bin_index];
            if remaining_bin_count == 0 {
                warn!("bin count was zero for assignment {a_index}, skipping");
                continue;
            }

            trace!("optimal_assign_big_alloc g7.b.loop.2.{a_index}");
            let mut random_pos = rng.gen_range(0..remaining_bin_count);
            for pipeline_index in 0..max_valid_pipelines {
                trace!("optimal_assign_big_alloc g7.b.loop.3.{a_index}.loop.{pipeline_index}");
                let count = assignment_counts[[bin_index, pipeline_index]];
                if count == 0 {
                    continue;
                }
                if random_pos < count {
                    remaining_bin_counts[bin_index] -= 1;
                    best_assignments.push(pipeline_index);
                    break;
                }
                random_pos -= count;
            }
            if best_assignments.len() != a_index + 1 {
                error!("no pipeline was found for assignment {a_index}, skipping");
                continue;
            }
            trace!("optimal_assign_big_alloc g7.b.loop.4.{a_index}");
        }

        trace!("optimal_assign_big_alloc g8.b");

        // check that the assignments are valid
        if let Err(e) = check_and_reset(budget, assignments, &best_assignments) {
            error!("check and reset error: {e:?}");
            return Err(OptimalAssignmentAlgorithmError::BudgetValidityError {
                mem_table: OptimalMemTable::Binwise(BinwiseMemTable {
                    mem_table,
                    num_slots,
                    num_bins,
                    num_pipelines: max_valid_pipelines,
                }),
                budget_error: e,
            });
        }
        trace!("optimal_assign_big_alloc g9.b");
        // assign all of them now that we know they're good
        for (a, &p) in assignments.iter_mut().zip(best_assignments.iter()) {
            a.pipeline = p;
        }

        trace!("optimal_assign_big_alloc g10.b");
        Ok(OptimalMemTable::Binwise(BinwiseMemTable {
            mem_table,
            num_slots,
            num_bins,
            num_pipelines: max_valid_pipelines,
        }))
    }
}

// we progress recursively through the dynamic view into the ndarray
// at each level, we may choose between the pipelines available to us, iterating through them,
// and then we store the cost and reward of the current assignment
fn optimal_assign_pipeline_per_item<Label>(
    current_solution_total_reward: f64,
    current_solution_total_cost: f64,
    total_budget: f64,
    assginments: &Vec<Assignment<Label>>,
    current_assignment: usize,
    current_assignment_pipeline: usize,
    // first dimension is the assignment, second dimension is the pipeline, third dimension is the cost and reward combination
    memoization_table: &mut ndarray::Array3<f64>,
    // bin_index_lookup: &HashMap<Option<Label>, usize>,
    // costs: &ndarray::Array2<f64>,
    // rewards: &ndarray::Array2<f64>,
    dims: &[usize],
    best_reward: &mut f64,
    // no need for best cost because best cost already terminated its recursion when it ran out of budget or ran out of items to allocate
) -> (f64, f64)
where
    Label: Debug + Clone + PartialEq + Eq + Hash,
{
    trace!("opt p^n g0 with dims {dims:?} which should be the same as the memoization table shape {:?}",  memoization_table.shape());
    const UNABLE_TO_CONTINUE: (f64, f64) = (f64::INFINITY, -1.0);
    const INDEX_DIM: usize = 0;
    const PIPELINE_DIM: usize = 1;
    // not needed because we directly write to the specific indices, of which there are only two
    // const QUALITY_DIM: usize = 2;

    // using COST_INDEX and REWARD_INDEX is clearer
    const COST_INDEX: usize = 0;
    const REWARD_INDEX: usize = 1;

    trace!("opt p^n g1, current assignment: {current_assignment}");
    if current_assignment >= dims[INDEX_DIM] {
        error!("current assignment slot index was greater than or equal to the number of items. make sure that you are recursing correctly, because we should have stopped before this point");
        return UNABLE_TO_CONTINUE;
    }

    trace!("opt p^n g2, current assignment pipeline: {current_assignment_pipeline}");
    if current_assignment_pipeline >= dims[PIPELINE_DIM] {
        error!("current assignment pipeline index was greater than or equal to the number of pipelines. make sure you're iterating over the possible pipelines correctly");
        return UNABLE_TO_CONTINUE;
    }

    // trace!("opt p^n g3");
    // let my_bin_id = &assginments[current_assignment].bin_info.id;
    // let my_bin_index = bin_index_lookup[my_bin_id];
    // let this_layer_cost = costs[[my_bin_index, current_assignment_pipeline]];
    // trace!("opt p^n g4");
    // let this_layer_reward = rewards[[my_bin_index, current_assignment_pipeline]];

    let this_layer_cost =
        assginments[current_assignment].bin_info.costs[current_assignment_pipeline];
    let this_layer_reward =
        assginments[current_assignment].bin_info.rewards[current_assignment_pipeline];

    trace!("opt p^n g5");
    if this_layer_reward < 0.0 {
        // we can't continue
        warn!("found negative layer reward with for assignment {current_assignment}, pipeline {current_assignment_pipeline}");
        warn!("current state of the algorithm is\n{memoization_table:#?}");
        return UNABLE_TO_CONTINUE;
    }

    trace!("opt p^n g6");
    // let current_index_best_cost = memoization_table[[current_assignment, current_assignment_pipeline, COST_INDEX]];
    let current_index_best_reward = memoization_table[[
        current_assignment,
        current_assignment_pipeline,
        REWARD_INDEX,
    ]];
    trace!("opt p^n g7");

    // info!("current assignment: {current_assignment}, current assignment pipeline: {current_assignment_pipeline}, current index best reward: {current_index_best_reward}, current solution total reward: {current_solution_total_reward}, current solution total cost: {current_solution_total_cost}, this layer cost: {this_layer_cost}, this layer reward: {this_layer_reward}");
    // info!("cost before: {current_solution_total_cost}");
    let new_cost = current_solution_total_cost + this_layer_cost;
    // info!("cost after: {new_cost}");
    if new_cost > total_budget {
        // don't even consider this further because it isn't valid to explore
        return UNABLE_TO_CONTINUE;
    }

    let new_reward = current_solution_total_reward + this_layer_reward;
    // handle if we're at the end of an assignment chain
    if current_assignment == dims[0] - 1 {
        trace!("opt p^n g8.term.0 - end of a chain - new reward: {new_reward}, current index best reward: {current_index_best_reward}");
        // this is it. either our new reward is better or we're done
        if new_reward > current_index_best_reward {
            trace!("opt p^n g8.term.1");
            // last second check for NAN
            if new_reward.is_nan() {
                error!("new reward was NAN");
                return UNABLE_TO_CONTINUE;
            }
            if new_cost.is_nan() {
                error!("new cost was NAN");
                return UNABLE_TO_CONTINUE;
            }
            memoization_table[[
                current_assignment,
                current_assignment_pipeline,
                REWARD_INDEX,
            ]] = new_reward;
            trace!("opt p^n g8.term.2");

            memoization_table[[current_assignment, current_assignment_pipeline, COST_INDEX]] =
                new_cost;
            trace!("opt p^n g8.term.3");

            if new_reward > *best_reward {
                *best_reward = new_reward;
            }

            return (new_cost, new_reward);
        } else {
            // we can't allow the earlier state to be corrupted by a previous potential good run that this index had,
            // so we disregard this expansion
            return UNABLE_TO_CONTINUE;
        }
    }
    trace!("opt p^n g8.continue");

    let current_assignment_bin = &assginments[current_assignment].bin_info;
    let valid_pipelines = &current_assignment_bin.valid_pipelines;
    trace!("opt p^n g9");
    // we aren't at the end, so we need to explore all possible pipelines that we might go down
    let mut local_best_reward = None;
    let mut local_best_cost = None;

    // we go through backwards so we prefer the later pipelines
    for &pipeline_index in valid_pipelines.iter().rev() {
        // for &pipeline_index in valid_pipelines.iter() {
        trace!("opt p^n g10.loop.{pipeline_index}.0");
        trace!(
            "trying assignment {} with pipeline {pipeline_index}",
            current_assignment + 1
        );
        let (new_local_cost, new_local_reward) = optimal_assign_pipeline_per_item(
            new_reward,
            new_cost,
            total_budget,
            assginments,
            current_assignment + 1,
            pipeline_index,
            memoization_table,
            // bin_index_lookup,
            // costs,
            // rewards,
            dims,
            best_reward,
        );
        trace!("opt p^n g10.loop.{pipeline_index}.1");
        if new_local_reward < 0.0 {
            // we can't continue
            trace!("new reward was negative, so we can't continue. this means a future pipeline was not valid, or it came up with a worse result than it found on another path");
            continue;
        }
        if new_local_reward > local_best_reward.unwrap_or(-1.0) {
            trace!("new local reward was better: {new_local_reward}");
            local_best_reward = Some(new_local_reward);
            local_best_cost = Some(new_local_cost);
        }
    }
    trace!("opt p^n g11");

    let Some(local_best_reward) = local_best_reward else {
        // warn!("local best reward was not found. this should not happen unless there were no valid pipelines");
        // if !valid_pipelines.is_empty() {
        //     error!("valid pipelines was not empty, but local best reward was not found. this should not happen");
        // }

        //  this actually means that *effectively* there were no valid pipelines because they would all be worse than the current best reward for those paths
        return UNABLE_TO_CONTINUE;
    };
    trace!("opt p^n g12");

    if local_best_reward > current_index_best_reward {
        trace!("opt p^n g13.better.0");
        memoization_table[[
            current_assignment,
            current_assignment_pipeline,
            REWARD_INDEX,
        ]] = local_best_reward;
        trace!("opt p^n g13.better.1");
        memoization_table[[current_assignment, current_assignment_pipeline, COST_INDEX]] =
            local_best_cost.expect(
                "local best cost was not found even though local best reward is being evaluated",
            );

        trace!("opt p^n g13.better.2");
        if local_best_reward > *best_reward {
            *best_reward = local_best_reward;
        }

        return (
            local_best_cost.expect(
                "local best cost was not found even though local best reward is being evaluated",
            ),
            local_best_reward,
        );
    } else {
        trace!("opt p^n g13.worse");
        // do not pursue this path
        return UNABLE_TO_CONTINUE;
    }
}

fn optimal_assign_count_per_pipeline_per_bin<Label>(
    current_solution_total_reward: f64,
    current_solution_total_cost: f64,
    total_budget: f64,
    bins: &Vec<BinInfo<Label>>,
    bin_counts: &[usize],
    current_bin_index: usize,
    current_assignment_pipeline: usize,
    current_assignment_amount: usize,
    current_bin_occupied_allocation: usize,
    total_occupied_allocation: usize,
    total_window_size: usize,
    // first dimension is the bin, second dimension is the pipeline, third dimension is amount to allot, fourth dimension is the cost and reward combination
    memoization_table: &mut ndarray::Array4<f64>,
    // costs: &ndarray::Array2<f64>,
    // rewards: &ndarray::Array2<f64>,
    dims: &[usize],
    best_reward: &mut f64,
    // no need for best cost because best cost already terminated its recursion when it ran out of budget or ran out of items to allocate
) -> (f64, f64)
where
    Label: Debug + Clone + PartialEq + Eq + Hash,
{
    const UNABLE_TO_CONTINUE: (f64, f64) = (f64::INFINITY, -1.0);
    const INDEX_DIM: usize = 0;
    const PIPELINE_DIM: usize = 1;
    const AMOUNT_DIM: usize = 1;
    const QUALITY_DIM: usize = 3;

    const COST_INDEX: usize = 0;
    const REWARD_INDEX: usize = 1;

    if current_bin_index >= dims[INDEX_DIM] {
        error!("current assignment slot index was greater than or equal to the number of items. make sure that you are recursing correctly, because we should have stopped before this point");
        return UNABLE_TO_CONTINUE;
    }

    if current_assignment_pipeline >= dims[PIPELINE_DIM] {
        error!("current assignment pipeline index was greater than or equal to the number of pipelines. make sure you're iterating over the possible pipelines correctly");
        return UNABLE_TO_CONTINUE;
    }
    if current_assignment_amount >= dims[AMOUNT_DIM] {
        error!("current assignment amount index was greater than or equal to the maximum number of items that a bin has+1. make sure you're iterating over the possible amounts correctly");
        return UNABLE_TO_CONTINUE;
    }

    let my_bin_count = bin_counts[current_bin_index];
    if current_assignment_amount > my_bin_count {
        error!("current assignment amount was greater than the number of items in the bin. this should not happen");
        return UNABLE_TO_CONTINUE;
    }

    // let this_layer_cost = costs[[current_bin_index, current_assignment_pipeline]];
    // let this_layer_reward = rewards[[current_bin_index, current_assignment_pipeline]];

    let this_layer_cost = bins[current_bin_index].costs[current_assignment_pipeline];
    let this_layer_reward = bins[current_bin_index].rewards[current_assignment_pipeline];

    if this_layer_reward < 0.0 {
        if current_assignment_amount > 0 {
            // we can't continue
            warn!("found negative layer reward with for nonzero assignment  of {current_assignment_amount} items in bin {current_bin_index}, pipeline {current_assignment_pipeline}");
            warn!("current state of the algorithm is\n{memoization_table:#?}");
            return UNABLE_TO_CONTINUE;
        }
        let remaining_items = total_window_size - total_occupied_allocation;
        if remaining_items == 0 {
            // we're done
            let current_best_reward = memoization_table[[
                current_bin_index,
                current_assignment_pipeline,
                current_assignment_amount,
                REWARD_INDEX,
            ]];
            if current_solution_total_reward > current_best_reward {
                memoization_table[[
                    current_bin_index,
                    current_assignment_pipeline,
                    current_assignment_amount,
                    REWARD_INDEX,
                ]] = current_solution_total_reward;
                memoization_table[[
                    current_bin_index,
                    current_assignment_pipeline,
                    current_assignment_amount,
                    COST_INDEX,
                ]] = current_solution_total_cost;
                if current_solution_total_reward > *best_reward {
                    *best_reward = current_solution_total_reward;
                }
                return (current_solution_total_cost, current_solution_total_reward);
            } else {
                return UNABLE_TO_CONTINUE;
            }
        }

        // we can't contribute anything, so just use this current reward and pass it to the next possible assignment
        let (next_bin_index, next_pipeline, next_bin_occupied_allocations) =
            if current_assignment_pipeline == dims[PIPELINE_DIM] - 1 {
                if current_bin_index == dims[INDEX_DIM] - 1 {
                    // we're at the end of the line
                    return (current_solution_total_cost, current_solution_total_reward);
                }
                // or we need to move to the next bin
                let bin = current_bin_index + 1;
                let pipeline = 0;
                (bin, pipeline, 0)
            } else {
                let bin = current_bin_index;
                let pipeline = current_assignment_pipeline + 1;
                (bin, pipeline, current_bin_occupied_allocation)
            };
        let remaining_budget = total_budget - current_solution_total_cost;
        let next_bin_count = bin_counts[next_bin_index];

        // let next_bin_cost = costs[[next_bin_index, next_pipeline]];
        let next_bin_cost = bins[next_bin_index].costs[next_pipeline];

        let mut next_bin_max_amount = next_bin_count - next_bin_occupied_allocations;
        // figure out how many we can allocate
        next_bin_max_amount =
            next_bin_max_amount.min((remaining_budget / next_bin_cost).floor() as usize);

        // let next_bin_reward = rewards[[next_bin_index, next_pipeline]];
        let next_bin_reward = bins[next_bin_index].rewards[next_pipeline];

        if next_bin_reward < 0.0 {
            next_bin_max_amount = 0;
        }

        let mut local_best_reward = None;
        let mut local_best_cost = None;

        for next_amount in 0..=next_bin_max_amount {
            let (new_local_cost, new_local_reward) = optimal_assign_count_per_pipeline_per_bin(
                current_solution_total_reward,
                current_solution_total_cost,
                total_budget,
                bins,
                bin_counts,
                next_bin_index,
                next_pipeline,
                next_amount,
                next_bin_occupied_allocations,
                total_occupied_allocation,
                total_window_size,
                memoization_table,
                // costs,
                // rewards,
                dims,
                best_reward,
            );
            if new_local_reward < 0.0 {
                // we can't continue
                trace!("new reward was negative, so we can't continue. this means a future pipeline was not valid, or it came up with a worse result than it found on another path");
                continue;
            }
            if new_local_reward > local_best_reward.unwrap_or(-f64::EPSILON) {
                local_best_reward = Some(new_local_reward);
                local_best_cost = Some(new_local_cost);
            }
        }

        let Some(local_best_reward) = local_best_reward else {
            error!("local best reward was not found when we had no valid pipelines left");
            return UNABLE_TO_CONTINUE;
        };

        if local_best_reward < 0.0 {
            // we can't continue
            trace!("local best reward was negative, so we can't continue. this means a future pipeline was not valid, or it came up with a worse result than it found on another path");
            return UNABLE_TO_CONTINUE;
        }

        let local_best_cost = local_best_cost.expect(
            "local best cost was not found even though local best reward is being evaluated",
        );

        let current_index_best_reward = memoization_table[[
            current_bin_index,
            current_assignment_pipeline,
            current_assignment_amount,
            REWARD_INDEX,
        ]];

        if local_best_reward > current_index_best_reward {
            memoization_table[[
                current_bin_index,
                current_assignment_pipeline,
                current_assignment_amount,
                REWARD_INDEX,
            ]] = local_best_reward;
            memoization_table[[
                current_bin_index,
                current_assignment_pipeline,
                current_assignment_amount,
                COST_INDEX,
            ]] = local_best_cost;

            if local_best_reward > *best_reward {
                *best_reward = local_best_reward;
            }

            return (local_best_cost, local_best_reward);
        } else {
            // do not pursue this path
            return UNABLE_TO_CONTINUE;
        }
    }

    let new_cost = current_solution_total_cost + this_layer_cost * current_assignment_amount as f64;
    let new_reward =
        current_solution_total_reward + this_layer_reward * current_assignment_amount as f64;
    if new_cost > total_budget {
        // don't even consider this further because it isn't valid to explore
        return UNABLE_TO_CONTINUE;
    }

    let new_assignment_amount = current_bin_occupied_allocation + current_assignment_amount;

    if new_assignment_amount > my_bin_count {
        error!("new assignment amount created a total allocation amount that was greater than the number of items in the bin. this should not happen");
        return UNABLE_TO_CONTINUE;
    }

    // handle if we're at the end of an assignment chain
    if current_bin_index == dims[INDEX_DIM] - 1
        && current_assignment_pipeline == dims[PIPELINE_DIM] - 1
    {
        // this is it. either our new reward is better or we're done
        if new_reward
            > memoization_table[[
                current_bin_index,
                current_assignment_pipeline,
                current_assignment_amount,
                REWARD_INDEX,
            ]]
        {
            // last second check for NAN
            if new_reward.is_nan() {
                error!("new reward was NAN");
                return UNABLE_TO_CONTINUE;
            }
            if new_cost.is_nan() {
                error!("new cost was NAN");
                return UNABLE_TO_CONTINUE;
            }
            memoization_table[[
                current_bin_index,
                current_assignment_pipeline,
                current_assignment_amount,
                REWARD_INDEX,
            ]] = new_reward;
            memoization_table[[
                current_bin_index,
                current_assignment_pipeline,
                current_assignment_amount,
                COST_INDEX,
            ]] = new_cost;

            if new_reward > *best_reward {
                *best_reward = new_reward;
            }

            return (new_cost, new_reward);
        } else {
            // we can't allow the earlier state to be corrupted by a previous potential good run that this index had,
            // so we disregard this expansion
            return UNABLE_TO_CONTINUE;
        }
    }

    let new_total_occupied_allocation = total_occupied_allocation + current_assignment_amount;

    // we aren't at the end, so we need to explore all possible pipelines that we might go down
    let (next_bin_index, next_pipeline_index, next_bin_occupied_allocation) =
        if current_assignment_pipeline == dims[PIPELINE_DIM] - 1 {
            if current_bin_index == dims[INDEX_DIM] - 1 {
                // we're at the end of the line
                return (new_cost, new_reward);
            }
            // otherwise we need to move to the next bin
            let bin = current_bin_index + 1;
            let pipeline = 0;
            (bin, pipeline, 0)
        } else {
            let bin = current_bin_index;
            let pipeline = current_assignment_pipeline + 1;
            (bin, pipeline, new_assignment_amount)
        };

    let remaining_budget = total_budget - new_cost;
    let next_bin_count = bin_counts[next_bin_index];

    // let next_bin_cost = costs[[next_bin_index, next_pipeline_index]];
    let next_bin_cost = bins[next_bin_index].costs[next_pipeline_index];

    let mut next_bin_max_amount = next_bin_count - next_bin_occupied_allocation;
    next_bin_max_amount =
        next_bin_max_amount.min((remaining_budget / next_bin_cost).floor() as usize);

    // let next_bin_reward = rewards[[next_bin_index, next_pipeline_index]];
    let next_bin_reward = bins[next_bin_index].rewards[next_pipeline_index];

    if next_bin_reward < 0.0 {
        next_bin_max_amount = 0;
    }

    let mut local_best_reward = None;
    let mut local_best_cost = None;

    for next_amount in 0..=next_bin_max_amount {
        let (new_local_cost, new_local_reward) = optimal_assign_count_per_pipeline_per_bin(
            new_reward,
            new_cost,
            total_budget,
            bins,
            bin_counts,
            next_bin_index,
            next_pipeline_index,
            next_amount,
            new_assignment_amount,
            new_total_occupied_allocation,
            total_window_size,
            memoization_table,
            // costs,
            // rewards,
            dims,
            best_reward,
        );
        if new_local_reward < 0.0 {
            // we can't continue
            trace!("new reward was negative, so we can't continue. this means a future pipeline was not valid, or it came up with a worse result than it found on another path");
            continue;
        }
        if new_local_reward > local_best_reward.unwrap_or(-f64::EPSILON) {
            local_best_reward = Some(new_local_reward);
            local_best_cost = Some(new_local_cost);
        }
    }

    let Some(local_best_reward) = local_best_reward else {
        error!("local best reward was not found when we had no valid pipelines left");
        return UNABLE_TO_CONTINUE;
    };

    if local_best_reward < 0.0 {
        // we can't continue
        trace!("local best reward was negative, so we can't continue. this means a future pipeline was not valid, or it came up with a worse result than it found on another path");
        return UNABLE_TO_CONTINUE;
    }

    let local_best_cost = local_best_cost
        .expect("local best cost was not found even though local best reward is being evaluated");

    let current_index_best_reward = memoization_table[[
        current_bin_index,
        current_assignment_pipeline,
        current_assignment_amount,
        REWARD_INDEX,
    ]];

    if local_best_reward > current_index_best_reward {
        memoization_table[[
            current_bin_index,
            current_assignment_pipeline,
            current_assignment_amount,
            REWARD_INDEX,
        ]] = local_best_reward;
        memoization_table[[
            current_bin_index,
            current_assignment_pipeline,
            current_assignment_amount,
            COST_INDEX,
        ]] = local_best_cost;

        if local_best_reward > *best_reward {
            *best_reward = local_best_reward;
        }

        return (local_best_cost, local_best_reward);
    } else {
        // do not pursue this path
        return UNABLE_TO_CONTINUE;
    }
}
pub trait LabelCategory: Clone + Debug + PartialEq + Eq + Hash {
    fn values() -> impl Iterator<Item = Self>;
}

#[cfg(test)]
#[test]
fn test_assignment() {
    use simple_logger::SimpleLogger;
    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .env()
        .init()
        .unwrap();
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    let mut overall_optimal_timings = HashMap::new();
    let mut avg_optimal_timings = Vec::new();
    let mut overall_greedy_timings = HashMap::new();
    let mut avg_greedy_timings = Vec::new();
    let mut overall_optimal_rewards = HashMap::new();
    let mut overall_greedy_rewards = HashMap::new();
    for num_assignments in 1usize..=12 {
        let mut window_optimal_timings = Vec::<f64>::new();
        let mut window_greedy_timings = Vec::<f64>::new();
        for seed_number in 0..1_000 {
            let mut rng = SmallRng::seed_from_u64(u64::MAX / (seed_number + 3));

            let mut assignments = vec![];
            // use BasicCategory as the Label for the assignments we make
            let a = Assignment {
                bin_info: time_series::HARD_CLASS_BIN,
                pipeline: 0,
            };
            let b = Assignment {
                bin_info: time_series::EASY_CLASS_BIN,
                pipeline: 0,
            };
            // let num_assignments = 5;
            let num_assignments = num_assignments;
            for _ in 0..num_assignments {
                if rng.gen_bool(0.5) {
                    assignments.push(a.clone());
                } else {
                    assignments.push(b.clone());
                }
            }

            let budget = num_assignments as f64
                * rng.gen_range(0.0..=1.0)
                * time_series::HARD_CLASS_BIN.costs.last().unwrap()
                // sometimes we want some extra budget to give them each the best chance
                * 1.2;

            // println!("budget is {budget} for {num_assignments} items");

            let mut assignments2 = assignments.clone();

            let opt_start_time = std::time::Instant::now();
            let opt_result = optimal_assign_big_alloc(budget, &mut assignments);
            // assert!(opt_result.is_ok(), "optimal assignment failed with error: {}");
            let opt_result = match opt_result {
                Ok(r) => r,
                Err(e) => {
                    panic!(
                        "optimal assignment for {} assignments with budget {} failed with error: {e:?}",
                        num_assignments, budget
                    );
                }
            };
            let opt_end_time = std::time::Instant::now();
            let opt_elapsed = opt_end_time - opt_start_time;
            window_optimal_timings.push(opt_elapsed.as_nanos() as f64);

            let mut total_cost = 0.0;
            let mut total_reward = 0.0;
            for a in &assignments {
                let pipeline_pos = a
                    .bin_info
                    .valid_pipelines
                    .iter()
                    .position(|x| *x == a.pipeline)
                    .unwrap();
                total_cost += a.bin_info.costs[pipeline_pos];
                total_reward += a.bin_info.rewards[pipeline_pos];
            }
            // println!("{assignments:?}");
            assert!(total_cost <= budget);
            let opt_reward = total_reward;

            // let mut swaps = Vec::new();
            let mut swap_classes = HashMap::new();
            let items_len = time_series::HARD_CLASS_BIN.costs.len();
            assert!(items_len > 1);
            for (i, j) in (0..items_len - 1).zip(1..items_len) {
                let i_cost = time_series::HARD_CLASS_BIN.costs[i];
                let j_cost = time_series::HARD_CLASS_BIN.costs[j];
                let i_reward = time_series::HARD_CLASS_BIN.rewards[i];
                let j_reward = time_series::HARD_CLASS_BIN.rewards[j];

                let i_score = if i_cost == 0.0 {
                    i_reward
                } else {
                    i_reward / i_cost
                };
                let j_score = if j_cost == 0.0 {
                    j_reward
                } else {
                    j_reward / j_cost
                };

                let swap_info = SwapInfo {
                    item_index: 0,
                    starting_pipeline: time_series::HARD_CLASS_BIN.valid_pipelines[i],
                    ending_pipeline: time_series::HARD_CLASS_BIN.valid_pipelines[j],
                    score_diff: j_score - i_score,
                    swap_cost: j_cost - i_cost,
                    reward_diff: j_reward - i_reward,
                    bin_info: time_series::HARD_CLASS_BIN.id,
                };
                swap_classes
                    .entry(time_series::HARD_CLASS_BIN.id)
                    .or_insert_with(Vec::new)
                    .push(swap_info);
            }

            let items_len = time_series::EASY_CLASS_BIN.costs.len();
            assert!(items_len > 1);
            for (i, j) in (0..items_len - 1).zip(1..items_len) {
                let i_cost = time_series::EASY_CLASS_BIN.costs[i];
                let j_cost = time_series::EASY_CLASS_BIN.costs[j];
                let i_reward = time_series::EASY_CLASS_BIN.rewards[i];
                let j_reward = time_series::EASY_CLASS_BIN.rewards[j];

                let i_score = if i_cost == 0.0 {
                    i_reward
                } else {
                    i_reward / i_cost
                };
                let j_score = if j_cost == 0.0 {
                    j_reward
                } else {
                    j_reward / j_cost
                };

                let swap_info = SwapInfo {
                    item_index: 0,
                    starting_pipeline: time_series::EASY_CLASS_BIN.valid_pipelines[i],
                    ending_pipeline: time_series::EASY_CLASS_BIN.valid_pipelines[j],
                    score_diff: j_score - i_score,
                    swap_cost: j_cost - i_cost,
                    reward_diff: j_reward - i_reward,
                    bin_info: time_series::EASY_CLASS_BIN.id,
                };
                swap_classes
                    .entry(time_series::EASY_CLASS_BIN.id)
                    .or_insert_with(Vec::new)
                    .push(swap_info);
            }

            let mut swaps = Vec::new();
            for (i, assignment) in assignments2.iter().enumerate() {
                let bin_swaps = swap_classes.get(&assignment.bin_info.id).unwrap();
                swaps.extend(bin_swaps.iter().cloned().map(|mut s| {
                    s.item_index = i;
                    s
                }));
            }

            let greedy_start_time = std::time::Instant::now();
            let greedy_result = greedy_assign(budget, &mut swaps, &mut assignments2);
            let greedy_end_time = std::time::Instant::now();
            let greedy_elapsed = greedy_end_time - greedy_start_time;
            window_greedy_timings.push(greedy_elapsed.as_nanos() as f64);

            let mut total_cost = 0.0;
            let mut total_reward = 0.0;
            for a in &assignments2 {
                let pipeline_pos: usize = a
                    .bin_info
                    .valid_pipelines
                    .iter()
                    .position(|x| *x == a.pipeline)
                    .unwrap();
                total_cost += a.bin_info.costs[pipeline_pos];
                total_reward += a.bin_info.rewards[pipeline_pos];
            }
            assert!(total_cost <= budget);
            let greedy_reward = total_reward;

            if opt_reward < greedy_reward {
                println!("greedy reward: {}", greedy_reward);
                println!("optimal reward: {}", opt_reward);
                println!("{:#?}", opt_result);
                println!("\n assignments\n{:#?}", assignments);
            }
            assert!(opt_reward >= greedy_reward);

            overall_greedy_rewards
                .entry(num_assignments)
                .or_insert(Vec::new())
                .push(greedy_reward);
            overall_optimal_rewards
                .entry(num_assignments)
                .or_insert(Vec::new())
                .push(opt_reward);
        }
        let avg_opt_time =
            window_optimal_timings.iter().sum::<f64>() / window_optimal_timings.len() as f64;
        let avg_greedy_time =
            window_greedy_timings.iter().sum::<f64>() / window_greedy_timings.len() as f64;
        overall_optimal_timings.insert(num_assignments, window_optimal_timings);
        overall_greedy_timings.insert(num_assignments, window_greedy_timings);
        avg_optimal_timings.push((num_assignments, avg_opt_time));
        avg_greedy_timings.push((num_assignments, avg_greedy_time));
    }

    let mut vals = Vec::new();
    for ((i, g), (_, o)) in avg_greedy_timings.iter().zip(avg_optimal_timings.iter()) {
        println!("average greedy time for {} assignments: {}", i, g);
        println!("average optimal time for {} assignments: {}", i, o);
        let mut map = serde_json::map::Map::new();
        map.insert("num_assignments".into(), (*i).into());
        map.insert(
            "greedy_rewards".into(),
            overall_greedy_rewards.remove(i).unwrap().into(),
        );
        map.insert(
            "optimal_rewards".into(),
            overall_optimal_rewards.remove(i).unwrap().into(),
        );
        map.insert(
            "greedy_timings_ns".into(),
            overall_greedy_timings.remove(i).into(),
        );
        map.insert(
            "optimal_timings_ns".into(),
            overall_optimal_timings.remove(i).into(),
        );
        vals.push(map);
    }

    println!("\n#DELIMITER#\n");
    println!("{}", serde_json::to_string(&vals).unwrap());
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub struct Bounded<const LOW: usize, const HIGH: usize>(usize);
impl<const LOW: usize, const HIGH: usize> LabelCategory for Bounded<LOW, HIGH> {
    fn values() -> impl Iterator<Item = Self> {
        (LOW..=HIGH).into_iter().map(Bounded)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum FutureWindowKind {
    TimeMillis(u128),
    Count(usize),
    TimeWithMaximumCount { time_ms: u128, max_count: usize },
}

pub mod basic_probability_forecast {
    use core::f64;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::{HashMap, VecDeque};
    use std::fmt::Debug;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
    pub enum BasicCategory {
        HardClass,
        EasyClass,
    }

    impl super::LabelCategory for BasicCategory {
        fn values() -> impl Iterator<Item = Self> {
            vec![BasicCategory::HardClass, BasicCategory::EasyClass].into_iter()
        }
    }

    use super::*;
    #[derive(Debug, Copy, Clone)]
    pub struct PendingData<L> {
        pub tuple_id: usize,
        pub category: Option<L>,
        pub time_of_creation_ns: u128,
        pub age_when_scheduling_ns: u128,
        pub time_of_scheduling: std::time::Instant,
    }
    #[derive(Debug, Copy, Clone)]
    pub struct PastData<L> {
        pub tuple_id: usize,
        pub category: Option<L>,
        pub time_of_creation_ns: u128,
        pub age_when_scheduling_ns: u128,
        pub time_of_scheduling: std::time::Instant,
        pub time_merged: std::time::Instant,
        pub time_elapsed_ms: f64,
        pub pipeline_id: usize,
    }

    #[derive(Debug, Clone)]
    pub struct History<L> {
        // rotating queue of the last n categorizations, along with how long they took to process
        pub(crate) past_data: VecDeque<PastData<L>>,
        // map from tuple id to the category it was placed in and the time it was created
        pub(crate) pending_data: nohash_hasher::IntMap<usize, PendingData<L>>,
        pub(crate) ingress_history: VecDeque<std::time::Instant>,
        pub(crate) back_channel:
            crossbeam::channel::Receiver<Vec<(usize, usize, std::time::Instant)>>,
        pub(crate) keep_n_history_items: usize,
        pub(crate) discrete_bins: Vec<BinInfo<L>>,
        pub(crate) bin_counts: HashMap<Option<L>, usize>,
        pub(crate) rng: SmallRng,
        // an extra function to account for more complex adjustments to the forecast beyond a time budget. We can forecast the basic categories, but we may need to adjust the reward or cost afterwards using custom past data
        pub(crate) adjust_forecast: Option<AdjusterFunction<L>>,
        pub exclude_purged_items_from_history: bool,
    }

    #[derive(Clone)]
    pub struct AdjusterFunction<L>(pub Arc<Mutex<dyn Send + FnMut(&mut [BinInfo<L>])>>);
    impl<L> Debug for AdjusterFunction<L> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("AdjusterFunction").finish()
        }
    }

    const ONE_MICROSECOND_AS_NANOS: f64 = 1_000.0;
    impl<L: LabelRequirements> History<L> {
        const MINIMUM_TIME_BETWEEN_INGRESS_NANOS: f64 = ONE_MICROSECOND_AS_NANOS;

        pub fn new(
            keep_n_history_items: usize,
            back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, std::time::Instant)>>,
            discrete_bins: Vec<BinInfo<L>>,
        ) -> Self {
            let bin_counts = discrete_bins
                .iter()
                .map(|bin| (bin.id.clone(), 0))
                .collect();
            History {
                past_data: Default::default(),
                pending_data: nohash_hasher::IntMap::with_capacity_and_hasher(
                    keep_n_history_items + 1,
                    Default::default(),
                ),
                back_channel,
                keep_n_history_items,
                ingress_history: VecDeque::new(),
                discrete_bins,
                bin_counts,
                rng: SmallRng::from_entropy(),
                adjust_forecast: None,
                exclude_purged_items_from_history: false,
            }
        }

        pub fn new_with_adjustment(
            keep_n_history_items: usize,
            back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, std::time::Instant)>>,
            discrete_bins: Vec<BinInfo<L>>,
            adjust_forecast: impl 'static + Send + FnMut(&mut [BinInfo<L>]),
        ) -> Self {
            let bin_counts = discrete_bins
                .iter()
                .map(|bin| (bin.id.clone(), 0))
                .collect();
            History {
                past_data: Default::default(),
                pending_data: nohash_hasher::IntMap::with_capacity_and_hasher(
                    keep_n_history_items + 1,
                    Default::default(),
                ),
                back_channel,
                keep_n_history_items,
                ingress_history: VecDeque::new(),
                discrete_bins,
                bin_counts,
                rng: SmallRng::from_entropy(),
                adjust_forecast: Some(AdjusterFunction(Arc::new(Mutex::new(adjust_forecast)))),
                exclude_purged_items_from_history: false,
            }
        }

        pub fn add_past_data(&mut self, past_data: PastData<L>) {
            debug!("adding past data for tuple id {}", past_data.tuple_id);
            let bin_id = past_data.category.clone();
            self.past_data.push_back(past_data);
            self.bin_counts
                .entry(bin_id)
                .and_modify(|e| *e += 1)
                .or_insert(1);
            self.past_data
                .make_contiguous()
                .sort_by(|a, b| a.time_of_scheduling.cmp(&b.time_of_scheduling));
            if self.past_data.len() > self.keep_n_history_items {
                let category = &self.past_data.front().unwrap().category;
                if let Some(bin_count) = self.bin_counts.get_mut(category) {
                    *bin_count -= 1;
                    self.past_data.pop_front();
                } else {
                    error!("bin count was not found for category {:?}", category);
                }
            }
        }

        pub fn record_ingress(&mut self, count: usize) {
            let now = std::time::Instant::now();
            for _ in 0..count {
                self.ingress_history.push_back(now);
            }
            self.ingress_history.make_contiguous().sort_unstable();
            while self.ingress_history.len() > self.keep_n_history_items {
                self.ingress_history.pop_front();
            }
        }

        pub fn send(
            &mut self,
            tuples: Vec<Tuple>,
            bins: Vec<BinInfo<L>>,
            mut target_pipe: usize,
            pipes: &[AsyncPipe],
        ) -> Result<(), AsyncPipeSendError> {
            use std::sync::LazyLock;
            #[derive(Debug, Copy, Clone, PartialEq, Eq)]
            enum RerouteZeroOption {
                Ignore,
                First,
                Random,
            }
            static REROUTE_ZERO_OPTION: LazyLock<RerouteZeroOption> = LazyLock::new(|| {
                let Ok(mut v) = std::env::var("REROUTE_ZERO_OPTION") else {
                    return RerouteZeroOption::Ignore;
                };
                v.make_ascii_lowercase();
                let v = v.trim();
                if v == "first" || v == "1" || v == "true" {
                    RerouteZeroOption::First
                } else if v == "random" || v == "rand" || v == "r" {
                    RerouteZeroOption::Random
                } else {
                    warn!("REROUTE_ZERO_OPTION was set to an unrecognized value: {}. valid values are First ('first', '1', 'true') or Random ('random', 'rand', 'r'). ignoring the variable and using pipe 0 as planned", v);
                    RerouteZeroOption::Ignore
                }
            });

            const AQUIFER_SPLIT_SEND_ENV_VAR: &str = "AQUIFER_SPLIT_SEND";
            const DEFAULT_SPLIT_SEND: bool = true;
            static SPLIT_SEND: LazyLock<bool> = LazyLock::new(|| {
                let Ok(mut v) = std::env::var(AQUIFER_SPLIT_SEND_ENV_VAR) else {
                    return DEFAULT_SPLIT_SEND;
                };
                v.make_ascii_lowercase();
                let v = v.trim();
                if v == "true" || v == "1" || v == "yes" {
                    true
                } else if v == "false" || v == "0" || v == "no" {
                    false
                } else {
                    warn!("{AQUIFER_SPLIT_SEND_ENV_VAR} was set to an unrecognized value: {}. valid values are True ('true', '1', 'yes') or False ('false', '0', 'no'). ignoring the variable and using the default value of {}", v, DEFAULT_SPLIT_SEND);
                    DEFAULT_SPLIT_SEND
                }
            });

            let now = std::time::Instant::now();
            self.record_ingress(tuples.len());
            let now_st = std::time::SystemTime::now();
            let now_unix_ns = match now_st.duration_since(std::time::UNIX_EPOCH) {
                Ok(v) => v.as_nanos(),
                Err(_) => {
                    error!("system time went backwards");
                    u128::MAX
                }
            };

            match (target_pipe, *REROUTE_ZERO_OPTION) {
                (0, RerouteZeroOption::Ignore) => {
                    self.send_helper_past_data_zeroes(&tuples, bins, now, now_unix_ns);
                }
                (0, RerouteZeroOption::First) => {
                    if pipes.len() > 1 {
                        warn!("rerouting {} tuples from pipe 0 to pipe 1 because REROUTE_ZERO_OPTION is set to First", tuples.len());
                        // return self.send(tuples, bins, 1, pipes);
                        target_pipe = 1;
                        self.send_helper_past_data(&tuples, bins, now, now_unix_ns);
                    } else {
                        warn!("REROUTE_ZERO_OPTION is set to First, but there is only one pipe (the zero/drop pipe). ignoring the variable and using pipe 0 as planned");
                        self.send_helper_past_data_zeroes(&tuples, bins, now, now_unix_ns);
                    }
                }
                (0, RerouteZeroOption::Random) => {
                    if pipes.len() > 1 {
                        let new_pipe = self.rng.gen_range(1..pipes.len());
                        warn!("rerouting from pipe 0 to pipe {} because REROUTE_ZERO_OPTION is set to Random", new_pipe);
                        // return self.send(tuples, bins, new_pipe, pipes);
                        target_pipe = new_pipe;
                        self.send_helper_past_data(&tuples, bins, now, now_unix_ns);
                    } else {
                        warn!("REROUTE_ZERO_OPTION is set to Random, but there is only one pipe (the zero/drop pipe). ignoring the variable and using pipe 0 as planned");
                        self.send_helper_past_data_zeroes(&tuples, bins, now, now_unix_ns);
                    }
                }
                // if it's not zero, then we don't care because it's not a drop that we would reroute
                (_, _) => {
                    self.send_helper_past_data(&tuples, bins, now, now_unix_ns);
                }
            }
            // TODO: if pending data+back channel is used, then this function will need to be reverted
            let tuple_ids: smallvec::SmallVec<[usize; 8]> =
                tuples.iter().map(|t| t.id() as usize).collect();
            if *SPLIT_SEND {
                for t in tuples {
                    pipes[target_pipe].send(vec![t])?;
                }
                Ok(())
            } else {
                match pipes[target_pipe].send(tuples) {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        // Cleanup pending data for inputs that failed to send
                        for id in tuple_ids {
                            self.pending_data.remove(&id);
                        }
                        Err(e)
                    }
                }
            }
        }

        pub fn cleanup_stale_pending(&mut self, safety_limit_ns: u128) {
            let now_unix_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();

            let mut to_remove = Vec::new();
            for (id, data) in self.pending_data.iter() {
                // If now < creation (clock skew?), result is 0, so not stale.
                // saturating_sub handles it.
                if now_unix_ns.saturating_sub(data.time_of_creation_ns) > safety_limit_ns {
                    to_remove.push((*id, data.clone()));
                }
            }
            if !to_remove.is_empty() {
                warn!(
                    "cleaning up {} stale pending items older than {} ns",
                    to_remove.len(),
                    safety_limit_ns
                );
                for (id, data) in to_remove {
                    self.pending_data.remove(&id);
                    // Record as failure to inform the scheduler of the overload
                    if !self.exclude_purged_items_from_history {
                        self.add_past_data(PastData {
                            tuple_id: id,
                            category: data.category,
                            age_when_scheduling_ns: data.age_when_scheduling_ns,
                            time_of_creation_ns: data.time_of_creation_ns,
                            time_of_scheduling: data.time_of_scheduling,
                            time_merged: std::time::Instant::now(),
                            // Record with max elapsed time allowed to signal failure
                            time_elapsed_ms: safety_limit_ns as f64 / 1_000_000.0,
                            pipeline_id: 0, // 0 indicates a drop/failure
                        });
                    }
                }
            }
        }

        fn send_helper_past_data(
            &mut self,
            tuples: &[Tuple],
            bins: Vec<BinInfo<L>>,
            now: std::time::Instant,
            now_unix_ns: u128,
        ) {
            self.pending_data
                .extend(tuples.iter().zip(bins).map(|(tuple, bin_info)| {
                    let tuple_id = tuple.id() as _;
                    let time_of_creation_ns = tuple.unix_time_created_ns();
                    let pending_data = PendingData {
                        tuple_id,
                        category: bin_info.id,
                        age_when_scheduling_ns: now_unix_ns - time_of_creation_ns,
                        time_of_scheduling: now,
                        time_of_creation_ns: tuple.unix_time_created_ns(),
                    };
                    (tuple_id, pending_data)
                }));
        }
        fn send_helper_past_data_zeroes(
            &mut self,
            tuples: &[Tuple],
            bins: Vec<BinInfo<L>>,
            now: std::time::Instant,
            now_unix_ns: u128,
        ) {
            // drop was not rerouted
            for (tuple, bin) in tuples.iter().zip(bins.into_iter()) {
                let time_of_creation_ns = tuple.unix_time_created_ns();
                let tuple_id = tuple.id() as _;
                let past_data = PastData {
                    tuple_id,
                    category: bin.id,
                    age_when_scheduling_ns: now_unix_ns - time_of_creation_ns,
                    time_of_scheduling: now,
                    time_merged: now,
                    time_elapsed_ms: 0.0,
                    time_of_creation_ns,
                    pipeline_id: 0,
                };
                self.add_past_data(past_data);
            }
        }

        pub fn update(&mut self) {
            trace!("history.update g0");
            // TODO: if the back channel is used, then this function may need to be reverted to reflect that.
            //   check git history to get the code for the while loop if necessary.
            self.ingress_history.make_contiguous().sort_unstable();

            let mut back_channel_count = 0;
            let mut matched_count = 0usize;
            let mut missing_count = 0usize;
            let mut matched_by_pipeline: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            let mut missing_by_pipeline: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            while let Ok(data) = self.back_channel.try_recv() {
                back_channel_count += data.len();
                for (tuple_id, pipeline_id, time_merged) in data {
                    if let Some(pending_data) = self.pending_data.remove(&tuple_id) {
                        matched_count += 1;
                        *matched_by_pipeline.entry(pipeline_id).or_default() += 1;
                        let past_data = PastData {
                            tuple_id,
                            category: pending_data.category,
                            age_when_scheduling_ns: pending_data.age_when_scheduling_ns,
                            time_of_creation_ns: pending_data.time_of_creation_ns,
                            time_of_scheduling: pending_data.time_of_scheduling,
                            time_merged,
                            time_elapsed_ms: (time_merged - pending_data.time_of_scheduling)
                                .as_nanos() as f64
                                / 1_000_000.0,
                            pipeline_id,
                        };
                        debug!("tuple id {} came back", tuple_id);
                        self.add_past_data(past_data);
                    } else {
                        missing_count += 1;
                        *missing_by_pipeline.entry(pipeline_id).or_default() += 1;
                        debug!("pending data was not found for tuple received from back channel with id {} (likely purged due to age)", tuple_id);
                    }
                }
            }
            if back_channel_count > 0 {
                info!(
                    "History updated: past_data={}, pending_data={}, processed {} feedback items (matched={}, missing={}), matched_by_pipeline={:?}, missing_by_pipeline={:?}",
                    self.past_data.len(),
                    self.pending_data.len(),
                    back_channel_count,
                    matched_count,
                    missing_count,
                    matched_by_pipeline,
                    missing_by_pipeline
                );
            }
        }

        pub fn mean_time_elapsed_ns(&self) -> Option<f64> {
            if self.past_data.is_empty() {
                return None;
            }
            Some(
                (self
                    .past_data
                    .iter()
                    .map(|item| item.time_elapsed_ms)
                    .sum::<f64>()
                    / (self.past_data.len() as f64))
                    * 1_000_000.0,
            )
        }

        pub fn mean_age_when_scheduling_ns(&self) -> Option<f64> {
            if self.past_data.is_empty() {
                return None;
            }
            Some(
                self.past_data
                    .iter()
                    .map(|item| item.age_when_scheduling_ns as f64)
                    .sum::<f64>()
                    / (self.past_data.len() as f64),
            )
        }

        pub fn mean_final_age_ns(&self) -> Option<f64> {
            if self.past_data.is_empty() {
                return None;
            }
            Some(
                self.past_data
                    .iter()
                    .map(|item| {
                        (item.age_when_scheduling_ns as f64) + (item.time_elapsed_ms * 1_000_000.0)
                    })
                    .sum::<f64>()
                    / (self.past_data.len() as f64),
            )
        }

        pub fn coarse_ingress_rate_ns_per_item(&self) -> Option<f64> {
            if true {
                if self.ingress_history.len() < 2 {
                    return None;
                }
                debug!(
                    "calculating coarse ingress rate with {} items in ingress history: {:?}",
                    self.ingress_history.len(),
                    self.ingress_history
                );
                let ns_diff = self
                    .ingress_history
                    .get(self.ingress_history.len() - 1)?
                    .saturating_duration_since(*self.ingress_history.get(0)?)
                    .as_nanos() as f64;
                let count = (self.ingress_history.len() - 1) as f64;
                let mut avg_ns = ns_diff / count;
                avg_ns = f64::max(avg_ns, Self::MINIMUM_TIME_BETWEEN_INGRESS_NANOS);
                return Some(avg_ns);
            }
            if self.past_data.is_empty() {
                return None;
            }
            let mut ages = Vec::with_capacity(self.past_data.len());
            let current_unix_time_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            for item in self.past_data.iter() {
                let age = (current_unix_time_ns - item.time_of_creation_ns) as f64;
                ages.push(age);
            }
            let oldest = ages.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let youngest = ages.iter().cloned().fold(f64::INFINITY, f64::min);
            // let average_time_per_item_micros =
            //     ((oldest - youngest) / self.past_data.len() as f64) / 1_000.0;
            // Some(1_000_000.0 / average_time_per_item_micros)
            let average_time_per_item_ns = (oldest - youngest) / self.past_data.len() as f64;
            Some(average_time_per_item_ns)
        }

        pub fn pending_count(&self) -> usize {
            self.pending_data.len()
        }

        pub fn mean_pending_age_ns(&self) -> Option<f64> {
            if self.pending_data.is_empty() {
                return None;
            }
            let current_unix_time_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let sum_age: u128 = self
                .pending_data
                .values()
                .map(|item| current_unix_time_ns.saturating_sub(item.time_of_creation_ns))
                .sum();
            Some(sum_age as f64 / self.pending_data.len() as f64)
        }

        pub fn max_pending_age_ns(&self) -> Option<f64> {
            if self.pending_data.is_empty() {
                return None;
            }
            let current_unix_time_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            self.pending_data
                .values()
                .map(|item| current_unix_time_ns.saturating_sub(item.time_of_creation_ns))
                .max()
                .map(|max_age| max_age as f64)
        }

        pub fn fine_ingress_rate_ns_per_item(&self) -> Option<f64> {
            if true {
                if self.ingress_history.len() < 2 {
                    return None;
                }
                debug!(
                    "calculating fine ingress rate with {} items in ingress history: {:?}",
                    self.ingress_history.len(),
                    self.ingress_history
                );
                let count = (self.ingress_history.len() - 1) as f64;
                let ns_diff_sum = self
                    .ingress_history
                    .iter()
                    .zip(self.ingress_history.iter().skip(1))
                    .map(|(a, b)| b.saturating_duration_since(*a).as_nanos() as f64)
                    .sum::<f64>();
                // return Some(ns_diff_sum / count);
                let mut avg_ns = ns_diff_sum / count;
                avg_ns = f64::max(avg_ns, Self::MINIMUM_TIME_BETWEEN_INGRESS_NANOS);
                return Some(avg_ns);
            }
            if self.past_data.len() < 2 {
                return None;
            }

            let mut diff_sum = 0.0;
            let num_diffs = self.past_data.len() - 1;
            for (a, b) in self.past_data.iter().zip(self.past_data.iter().skip(1)) {
                let Some(diff) = b
                    .time_of_scheduling
                    .checked_duration_since(a.time_of_scheduling)
                else {
                    error!(
                        "time went backwards between tuples {} and {}",
                        a.tuple_id, b.tuple_id
                    );
                    return None;
                };
                diff_sum += diff.as_nanos() as f64;
            }
            Some(diff_sum / num_diffs as f64)
        }

        pub fn mean_time_elapsed_per_item_ns(&self) -> Option<f64> {
            if self.past_data.is_empty() {
                return None;
            }
            Some(
                (self
                    .past_data
                    .iter()
                    .map(|item| item.time_elapsed_ms)
                    .sum::<f64>()
                    / (self.past_data.len() as f64))
                    * 1_000_000.0,
            )
        }

        // how much is elapsed going up on average between items in the history?
        // pub fn mean_elapsed_increase(&self) -> f64 {
        //     let mut elapsed_sorted_by_creation_time =
        //         self.past_data.iter().map(|item| (item.time_elapsed_ms, item.time_of_creation_ns)).collect::<Vec<_>>();
        //     elapsed_sorted_by_creation_time.sort_by(|a, b| a.1.cmp(&b.1));
        //     if elapsed_sorted_by_creation_time.len() < 2 {
        //         return 0.0;
        //     }
        //     let total_increase = elapsed_sorted_by_creation_time
        //         .iter()
        //         .zip(elapsed_sorted_by_creation_time.iter().skip(1))
        //         .map(|((a_elapsed, a_creation), (b_elapsed, b_creation))| {
        //             if a_creation >= b_creation {
        //                 error!("creation time went backwards between items with elapsed times {a_elapsed} and {b_elapsed}");
        //                 return 0.0;
        //             }
        //             *b_elapsed - *a_elapsed
        //         })
        //         .sum::<f64>();
        //     total_increase / (elapsed_sorted_by_creation_time.len() - 1) as f64
        // }

        // how much is elapsed going up on average between items in the history?
        pub fn mean_elapsed_increase_ms(&self) -> f64 {
            if self.past_data.len() < 2 {
                return 0.0;
            }
            let total_increase = self
                .past_data
                .iter()
                .zip(self.past_data.iter().skip(1))
                .map(|(a, b)| b.time_elapsed_ms - a.time_elapsed_ms)
                .sum::<f64>();
            total_increase / (self.past_data.len() - 1) as f64
        }

        // pub fn recent_weighted_mean_elapsed_increase_ms(&self) -> f64 {
        //     if self.past_data.len() < 2 {
        //         return 0.0;
        //     }
        //     let total_increase = self.past_data
        //         .iter()
        //         .zip(self.past_data.iter().skip(1))
        //         .map(|(a,b)| {
        //             b.time_elapsed_ms - a.time_elapsed_ms
        //         })
        //         .enumerate()
        //         .map(|(i, increase)| {
        //             // weight the increase by the index, so that more recent increases are weighted more heavily
        //             increase * ((i+1) as f64)
        //         })
        //         .sum::<f64>();
        //     // let weights_for_denominator = (1..=self.past_data.len() - 1)
        //     //     .map(|i| i as f64)
        //     //     .sum::<f64>();
        //     // closed form solution
        //     let weights_for_denominator = (self.past_data.len() - 1) as f64 * (self.past_data.len() as f64) / 2.0;
        //     // total_increase / (self.past_data.len() - 1) as f64
        //     total_increase / weights_for_denominator
        // }
        pub fn recent_weighted_mean_elapsed_increase_ms(&self) -> f64 {
            if self.past_data.len() < 2 {
                return 0.0;
            }
            Self::sequence_position_weighted_mean(
                self.past_data
                    .iter()
                    .zip(self.past_data.iter().skip(1))
                    .map(|(a, b)| b.time_elapsed_ms - a.time_elapsed_ms),
            )
        }
        pub fn sequence_position_weighted_mean(iter: impl Iterator<Item = f64>) -> f64 {
            let mut total = 0.0;
            let mut pos = 1.0;
            for value in iter {
                total += value * pos;
                pos += 1.0;
            }
            // let weight_sum = (1..=pos as usize - 1).map(|i| i as f64).sum::<f64>();
            // closed form solution
            let weight_sum = (pos as f64 * (pos - 1.0)) / 2.0;
            if weight_sum == 0.0 {
                return 0.0;
            }
            total / weight_sum
        }

        pub fn mean_creation_time_ns(&self) -> Option<f64> {
            if self.past_data.is_empty() {
                return None;
            }
            Some(
                self.past_data
                    .iter()
                    .map(|item| item.time_of_creation_ns as f64)
                    .sum::<f64>()
                    / (self.past_data.len() as f64),
            )
        }

        pub fn mean_age_when_merging_increase_ms(&self) -> f64 {
            if self.past_data.len() < 2 {
                return 0.0;
            }
            let total_increase = self
                .past_data
                .iter()
                .zip(self.past_data.iter().skip(1))
                .map(|(a, b)| {
                    // ((b.age_when_scheduling_ns as f64 / 1_000_000.0) + b.time_elapsed_ms as f64)
                    //     - ((a.age_when_scheduling_ns as f64 / 1_000_000.0)
                    //         + a.time_elapsed_ms as f64)
                    // a.time_merged.duration_since(a.time_of_scheduling).as_nanos() as f64 / 1_000_000.0
                    //     - b.time_merged.duration_since(b.time_of_scheduling).as_nanos() as f64
                    //         / 1_000_000.0

                    let a_age_when_merging_ms =
                        (a.age_when_scheduling_ns as f64 / 1_000_000.0) + a.time_elapsed_ms;
                    let b_age_when_merging_ms =
                        (b.age_when_scheduling_ns as f64 / 1_000_000.0) + b.time_elapsed_ms;
                    b_age_when_merging_ms - a_age_when_merging_ms
                })
                .sum::<f64>();
            total_increase / (self.past_data.len() - 1) as f64
        }

        pub fn recent_weighted_mean_age_when_merging_increase_ms(&self) -> f64 {
            if self.past_data.len() < 2 {
                return 0.0;
            }
            Self::sequence_position_weighted_mean(
                self.past_data
                    .iter()
                    .zip(self.past_data.iter().skip(1))
                    .map(|(a, b)| {
                        // ((b.age_when_scheduling_ns as f64 / 1_000_000.0) + b.time_elapsed_ms as f64)
                        //     - ((a.age_when_scheduling_ns as f64 / 1_000_000.0)
                        //         + a.time_elapsed_ms as f64)
                        let a_age_when_merging_ms =
                            (a.age_when_scheduling_ns as f64 / 1_000_000.0) + a.time_elapsed_ms;
                        let b_age_when_merging_ms =
                            (b.age_when_scheduling_ns as f64 / 1_000_000.0) + b.time_elapsed_ms;
                        // FIFO order, b is the more recent item
                        b_age_when_merging_ms - a_age_when_merging_ms
                    }),
            )
        }

        pub fn forecast_fn() -> impl Send
               + Sync
               + Clone
               + Fn(&mut Self, &[BinInfo<L>], FutureWindowKind) -> (Vec<BinInfo<L>>, f64)
        where
            L: LabelRequirements + Send + Sync + 'static,
        {
            let forecast_fn_inner = Self::forecast_fn_inner();
            move |this, known_bins, future_kind| forecast_fn_inner(this, known_bins, future_kind)
        }

        pub fn forecast_fn_inner() -> Arc<
            dyn Send
                + Sync
                + Fn(&mut Self, &[BinInfo<L>], FutureWindowKind) -> (Vec<BinInfo<L>>, f64),
        >
        where
            L: LabelRequirements + Send + Sync + 'static,
        {
            fn default_stride() -> usize {
                1
            }
            #[derive(Debug, Serialize, Deserialize)]
            struct StaticForecast<L> {
                all_bins: Vec<BinInfo<L>>,
                current_start_index: std::sync::atomic::AtomicUsize,
                // window_size: usize,
                #[serde(default = "default_stride")]
                stride: usize,
            }
            if let Ok(perfect_knowledge_env_file) = std::env::var("FORECAST_FILE") {
                let file = std::fs::File::open(perfect_knowledge_env_file)
                    .expect("failed to open perfect knowledge env file");
                let mut reader = std::io::BufReader::new(file);
                let static_forecast: StaticForecast<L> = serde_json::from_reader(&mut reader)
                    .expect("failed to deserialize perfect knowledge env file");
                let window_size_totals_storage = std::sync::atomic::AtomicUsize::new(0);
                let total_calls_storage = std::sync::atomic::AtomicUsize::new(0);
                return Arc::new(move |this, known_bins, future_kind| {
                    this.update();
                    let ns_per_item = this
                        .coarse_ingress_rate_ns_per_item()
                        .unwrap_or(1_000_000.0);
                    debug!("ns_per_item: {}", ns_per_item);
                    let (mut number_of_items, mut skipped_items) = match future_kind {
                        FutureWindowKind::TimeMillis(time_millis) => {
                            let ms_per_item = ns_per_item / 1_000_000.0;
                            let items_during_time = f64::max(1.0, time_millis as f64 / ms_per_item);
                            (items_during_time as usize, 0)
                        }
                        FutureWindowKind::Count(count) => (count, 0),
                        FutureWindowKind::TimeWithMaximumCount {
                            time_ms: time_millis,
                            max_count,
                        } => {
                            let ms_per_item = ns_per_item / 1_000_000.0;
                            let items_during_time =
                                f64::max(1.0, time_millis as f64 / ms_per_item) as usize;

                            let allowed_items = items_during_time.min(max_count);
                            let skipped_items = items_during_time.saturating_sub(allowed_items);
                            (allowed_items, skipped_items)
                        }
                    };
                    skipped_items = skipped_items.saturating_sub(known_bins.len());
                    let kept_percentage = if number_of_items > 0 {
                        1.0 - (skipped_items as f64 / number_of_items as f64)
                    } else {
                        1.0
                    };
                    number_of_items = number_of_items.saturating_sub(known_bins.len());

                    debug!(
                        "number_of_items: {} (+ {} current items)",
                        number_of_items,
                        known_bins.len()
                    );
                    let mut window_size_total =
                        window_size_totals_storage.load(std::sync::atomic::Ordering::Relaxed);
                    let total_calls =
                        total_calls_storage.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let window_size_avg =
                        f64::max(1.0, window_size_total as f64 / (total_calls as f64 + 1e-6));
                    debug!(
                        "window_size_avg: {}, total_calls: {}, window_size_total: {}",
                        window_size_avg, total_calls, window_size_total
                    );
                    window_size_total += number_of_items;
                    window_size_totals_storage
                        .store(window_size_total, std::sync::atomic::Ordering::Relaxed);

                    let mut forecast =
                        Vec::with_capacity(usize::max(window_size_avg as usize, number_of_items));

                    // we at least need to return the known bins, so we can start with them
                    forecast.extend_from_slice(known_bins);
                    if number_of_items == 0 {
                        warn!("number_of_items is 0, this should not happen, returning one-item forecast because we must progress");
                        static_forecast
                            .current_start_index
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        forecast.push(static_forecast.all_bins[0].clone());
                        return (forecast, kept_percentage);
                    }

                    let start_idx = static_forecast
                        .current_start_index
                        .load(std::sync::atomic::Ordering::Relaxed);

                    if number_of_items >= static_forecast.all_bins.len() {
                        warn!("number_of_items {number_of_items} is greater than the total number of bins {}, returning all bins, repeated enough times to make a full forecast", static_forecast.all_bins.len());
                        let mut item_iter =
                            static_forecast
                                .all_bins
                                .iter()
                                .cycle()
                                .skip(<usize as Ord>::clamp(
                                    start_idx,
                                    0,
                                    static_forecast.all_bins.len() - 1,
                                ));
                        static_forecast.current_start_index.store(
                            (start_idx + 1) % static_forecast.all_bins.len(),
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        while forecast.len() < number_of_items {
                            if let Some(item) = item_iter.next() {
                                forecast.push(item.clone());
                            } else {
                                error!("cycling iterator returned None, this should not happen");
                                break;
                            }
                        }
                        return (forecast, kept_percentage);
                    }
                    // so now we know that 0 < number_of_items < all_bins.len()
                    // does the current start index provide enough items to satisfy number_of_items?
                    let remaining_items = static_forecast.all_bins.len().saturating_sub(start_idx);
                    if remaining_items <= number_of_items {
                        debug!(
                            "remaining items {} are less than or equal to number_of_items {}, using the last {} items",
                            remaining_items, number_of_items, remaining_items
                        );
                        // use the last number_of_items items
                        let start_idx = static_forecast
                            .all_bins
                            .len()
                            .saturating_sub(number_of_items);
                        forecast.extend_from_slice(&static_forecast.all_bins[start_idx..]);
                        static_forecast.current_start_index.store(
                            static_forecast.all_bins.len() - (window_size_avg as usize),
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        return (forecast, kept_percentage);
                    }
                    // we have enough items from start_idx to satisfy number_of_items
                    forecast.extend_from_slice(
                        &static_forecast.all_bins[start_idx..(start_idx + number_of_items)],
                    );
                    static_forecast.current_start_index.store(
                        usize::min(
                            static_forecast.stride + start_idx,
                            static_forecast.all_bins.len(),
                        ),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    return (forecast, kept_percentage);
                });
            }

            Arc::new(move |this, known_bins, future_kind| {
                trace!("forecast_fn g0");
                this.update();
                trace!("forecast_fn g1");
                // if we don't have enough history to make a forecast, return the known bins
                let history_len = this.past_data.len();
                if history_len == 0 || history_len == 1 {
                    return (known_bins.to_vec(), 1.0);
                }
                trace!("forecast_fn g2");
                debug!("history: {:?}", this.past_data);
                let current_unix_time_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();

                let mut creation_times_ns = Vec::with_capacity(history_len);
                let mut ages_ns = Vec::with_capacity(history_len);
                let mut tuple_ids = Vec::with_capacity(history_len);
                for item in this.past_data.iter() {
                    ages_ns.push(
                        item.age_when_scheduling_ns as f64 + (item.time_elapsed_ms * 1_000_000.0),
                    );
                    creation_times_ns.push(item.time_of_creation_ns);
                    tuple_ids.push(item.tuple_id);
                }
                ages_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
                creation_times_ns.sort();
                tuple_ids.sort();

                let creation_time_diff_sum = creation_times_ns
                    .iter()
                    .zip(creation_times_ns.iter().skip(1))
                    .map(|(a, b)| b - a)
                    .sum::<u128>();
                let creation_time_diff_avg_nanos = f64::max(
                    creation_time_diff_sum as f64 / ((history_len - 1) as f64),
                    Self::MINIMUM_TIME_BETWEEN_INGRESS_NANOS,
                );
                // let oldest = ages_ns[0];
                // let youngest = ages_ns[history_len - 1];
                let oldest = creation_times_ns[0];
                let youngest = creation_times_ns[history_len - 1];
                let average_time_per_item_micros = creation_time_diff_avg_nanos as f64 / 1_000.0;
                let coarse_time_per_item_micros =
                    ((youngest - oldest) as f64 / history_len as f64) / 1_000.0;
                debug!("\nforecast_fn ages: {ages_ns:?},\ntuples: {tuple_ids:?}\ncreation times: {creation_times_ns:?}\noldest: {oldest},\nyoungest: {youngest}",);
                debug!("average individual time diff: {average_time_per_item_micros} micros, coarse time per item: {coarse_time_per_item_micros} micros");
                let average_time_per_item_micros = coarse_time_per_item_micros;

                // let average_time_per_item_micros =
                //     ((oldest - youngest) / history_len as f64) / 1_000.0;
                // let (Some(earliest_scheduling_time), Some(latest_scheduling_time)) =
                //     (earliest_creation_time, latest_scheduling_time)
                // else {
                //     error!("earliest or latest scheduling time was not found");
                //     return (known_bins.to_vec(), 1.0);
                // };
                // let average_time_per_item_micros =
                //     (latest_scheduling_time - earliest_scheduling_time).as_micros() as f64
                //         / history_len as f64;

                let average_time_per_item_micros = this
                    .coarse_ingress_rate_ns_per_item()
                    .unwrap_or(average_time_per_item_micros * 1_000.0)
                    / 1_000.0;

                trace!("forecast_fn g5");
                let mut unclipped_v = None;
                let num_future_items = match future_kind {
                    FutureWindowKind::TimeMillis(time_millis) => {
                        if average_time_per_item_micros < 100.0 {
                            warn!("average_time_per_item_micros was less than 100.0 in forecast_fn with average_time_per_item_micros: {average_time_per_item_micros} and a history of {history_len} items. setting to be {} items instead of dividing by a very small number", known_bins.len());
                            return (known_bins.to_vec(), 1.0);
                        } else {
                            let v = ((time_millis * 1_000) as f64 / average_time_per_item_micros)
                                as usize;
                            unclipped_v = Some(v as f64);
                            debug!("time_millis was {time_millis}, average_time_per_item_micros was {average_time_per_item_micros}, history_len was {history_len}, so we are forecasting {v} future items");
                            v
                        }
                    }
                    FutureWindowKind::TimeWithMaximumCount {
                        time_ms: time_millis,
                        max_count,
                    } => {
                        if average_time_per_item_micros < 100.0 {
                            warn!("average_time_per_item_micros was less than 100.0 in forecast_fn with average_time_per_item_micros: {average_time_per_item_micros} and a history of {history_len} items. setting to be {} items instead of dividing by a very small number", known_bins.len());
                            return (known_bins.to_vec(), 1.0);
                        } else {
                            let v = ((time_millis * 1_000) as f64 / average_time_per_item_micros)
                                as usize;
                            unclipped_v = Some(v as f64);
                            debug!("time_millis was {time_millis}, average_time_per_item_micros was {average_time_per_item_micros}, history_len was {history_len}, so we are forecasting {v} future items");
                            if v > max_count {
                                max_count
                            } else {
                                v
                            }
                        }
                    }
                    FutureWindowKind::Count(count) => count,
                };
                debug!(
                    "forecasting {} future items with a history of {} items",
                    num_future_items, history_len
                );

                trace!("forecast_fn g7");
                let mut future_bins = Vec::with_capacity(num_future_items.max(known_bins.len()));
                for known_bin in known_bins.iter() {
                    let future_bin = known_bin.clone();
                    future_bins.push(future_bin);
                }
                trace!("forecast_fn g8");
                let current_len = future_bins.len();
                if current_len >= num_future_items {
                    let final_ratio_kept =
                        unclipped_v.map_or(1.0, |v| (current_len as f64 / v).min(1.0));
                    return (future_bins, final_ratio_kept);
                }
                let amount_to_go = num_future_items - current_len;
                for _amt in 0..amount_to_go {
                    trace!("forecast_fn g9-{_amt}");
                    // select a bin from this.discrete_bins with the weight of the bin_counts
                    let random_float = this.rng.gen::<f64>();
                    let mut assigned_bin = &this.discrete_bins[0];
                    let mut i = 0;
                    for bin in this.discrete_bins.iter() {
                        trace!("forecast_fn g9-{_amt}_g10-{i}");
                        i += 1;
                        let bin_count = this.bin_counts.get(&bin.id).unwrap();
                        trace!("forecast_fn g9-{_amt}_g11-{i}");
                        let bin_weight = *bin_count as f64 / history_len as f64;
                        trace!("forecast_fn g9-{_amt}_g12-{i}");
                        if random_float < bin_weight {
                            assigned_bin = bin;
                            break;
                        }
                    }
                    trace!("forecast_fn g13-{_amt}");
                    future_bins.push(assigned_bin.clone());
                }
                if let Some(adjust_forecast) = &mut this.adjust_forecast {
                    let pushed_portion = &mut future_bins[current_len..];
                    let adjust_forecast = &mut adjust_forecast.0.lock().unwrap();
                    adjust_forecast(pushed_portion);
                }
                trace!("forecast_fn g14");
                let final_ratio_kept =
                    unclipped_v.map_or(1.0, |v| (future_bins.len() as f64 / v).min(1.0));

                // ADDED LOGGING FOR DEPLOYMENT DEBUGGING
                if let Some(v) = unclipped_v {
                    info!("forecast_fn: history_len={}, known_bins_len={}, unclipped_v={:.2}, future_bins_len={}, final_ratio_kept={:.4}. clipped={}", 
                        history_len, known_bins.len(), v, future_bins.len(), final_ratio_kept, v > future_bins.len() as f64);
                }

                (future_bins, final_ratio_kept)
            })
        }
    }

    #[cfg(test)]
    #[test]
    fn test_sequence_position_weighted_mean() {
        let values = vec![2.0, 2.0, 3.0, 4.0, 5.0];
        let normal_mean = values.iter().sum::<f64>() / values.len() as f64;
        let weighted_mean =
            History::<BasicCategory>::sequence_position_weighted_mean(values.iter().cloned());
        assert!(weighted_mean > normal_mean, "weighted mean {weighted_mean} should be greater than normal arithmetic mean {normal_mean}");
        println!(
            "weighted mean: {}, normal mean: {}",
            weighted_mean, normal_mean
        );
    }
}

// just checking that it typechecks
#[cfg(test)]
fn test_binning_function(t: &Tuple) -> BinInfo<basic_probability_forecast::BasicCategory> {
    let category = match t.id() % 2 {
        0 => basic_probability_forecast::BasicCategory::HardClass,
        1 => basic_probability_forecast::BasicCategory::EasyClass,
        _ => panic!("modulus of 2 was not 0 or 1"),
    };
    BinInfo {
        id: Some(category),
        valid_pipelines: (&[0, 1][..]).into(),
        rewards: (&[1.0, 0.5][..]).into(),
        costs: (&[0.5, 0.25][..]).into(),
    }
}

pub mod time_series {
    use super::basic_probability_forecast::*;
    use super::*;

    // old avgs
    const BIG_MODEL_RUNTIME: f64 = 60.8;
    const SMALL_MODEL_RUNTIME: f64 = 37.2;

    // new mean
    // const BIG_MODEL_RUNTIME: f64 = 62981440.17920782 / 1_000_000.0;
    // const SMALL_MODEL_RUNTIME: f64 = 48981347.71142238 / 1_000_000.0;

    // new 95th percentile
    // const BIG_MODEL_RUNTIME: f64 = 64102000.00000001 / 1_000_000.0;
    // const SMALL_MODEL_RUNTIME: f64 = 53463000.0 / 1_000_000.0;

    // new 99th percentile
    // const BIG_MODEL_RUNTIME: f64 = 66550660.0 / 1_000_000.0;
    // const SMALL_MODEL_RUNTIME: f64 = 72522000.0 / 1_000_000.0;

    pub(crate) const BIG_MODEL_REWARD: f64 = 0.82;
    pub(crate) const SMALL_MODEL_REWARD_HARD_CLASS: f64 = BIG_MODEL_REWARD
        * ((
            0.63 + // class 0 relative accuracy
            0.47 + // class 1 relative accuracy
            0.44 + // class 4 relative accuracy
            0.47
            // class 7 relative accuracy
        ) / 4.0);
    pub(crate) const SMALL_REWARD_EASY_CLASS: f64 = BIG_MODEL_REWARD; // in reality this is 1.004 * BIG_REWARD but we prefer the big model anyway

    pub(crate) const HARD_CLASS_BIN: BinInfo<BasicCategory> = BinInfo {
        id: Some(BasicCategory::HardClass),
        valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2]),
        rewards: ShareableArray::Borrowed(&[0.0, SMALL_MODEL_REWARD_HARD_CLASS, BIG_MODEL_REWARD]),
        costs: ShareableArray::Borrowed(&[0.0, SMALL_MODEL_RUNTIME, BIG_MODEL_RUNTIME]),
    };
    pub(crate) const EASY_CLASS_BIN: BinInfo<BasicCategory> = BinInfo {
        id: Some(BasicCategory::EasyClass),
        valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2]),
        rewards: ShareableArray::Borrowed(&[0.0, SMALL_REWARD_EASY_CLASS, BIG_MODEL_REWARD]),
        costs: ShareableArray::Borrowed(&[0.0, SMALL_MODEL_RUNTIME, BIG_MODEL_RUNTIME]),
    };
}

// This exists just to check types
#[cfg(test)]
mod tests {
    use super::basic_probability_forecast::*;
    use super::time_series::*;
    use super::*;
    #[allow(unused)]
    fn apply_test_binning_function(
        inputs: Vec<Tuple>,
        pipelines: &[AsyncPipe],
        history: &mut History<basic_probability_forecast::BasicCategory>,
        budget: f64,
    ) -> Option<usize> {
        aquifer_scheduler(
            inputs,
            pipelines,
            history,
            AlgInputs {
                binning_function: test_binning_function,
                forecast_function: History::forecast_fn(),
                send_function: History::send,
            },
            budget,
            FutureWindowKind::TimeMillis(budget as _),
        )
    }

    #[test]
    fn test_forecast() {
        // ... (existing code omitted for brevity in instruction, but I will replace the whole block)
        // unused channel. we will be putting items into history manually
        let (_tx, rx) = crossbeam::channel::unbounded();
        let mut history =
            basic_probability_forecast::History::new(10, rx, vec![HARD_CLASS_BIN, EASY_CLASS_BIN]);
        // test double-now
        let now = std::time::Instant::now();
        let _now_unix_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        history.past_data.push_back(PastData {
            tuple_id: 0,
            category: Some(BasicCategory::HardClass),
            age_when_scheduling_ns: 0,
            time_of_scheduling: now,
            time_of_creation_ns: 0,
            time_merged: now,
            time_elapsed_ms: 0.0,
            pipeline_id: 0,
        });
        history.past_data.push_back(PastData {
            tuple_id: 1,
            category: Some(BasicCategory::HardClass),
            age_when_scheduling_ns: 0,
            time_of_scheduling: now,
            time_of_creation_ns: 0,
            time_merged: now,
            time_elapsed_ms: 0.0,
            pipeline_id: 1,
        });
    }

    #[test]
    fn test_budget_normalization_large_batch() {
        let (_tx, rx) = crossbeam::channel::unbounded();
        let mut history = basic_probability_forecast::History::new(
            100,
            rx,
            vec![time_series::HARD_CLASS_BIN, time_series::EASY_CLASS_BIN],
        );

        // Fill history to get a stable ingress rate
        let now = std::time::Instant::now();
        let interval_ns = 1_000_000u128; // 1ms per item -> 1000 items per second
        for i in 0..50 {
            history
                .past_data
                .push_back(basic_probability_forecast::PastData {
                    tuple_id: i,
                    category: Some(basic_probability_forecast::BasicCategory::HardClass),
                    age_when_scheduling_ns: 0,
                    time_of_scheduling: now
                        + std::time::Duration::from_nanos((i as u64) * (interval_ns as u64)),
                    time_of_creation_ns: (i as u128) * interval_ns,
                    time_merged: now
                        + std::time::Duration::from_nanos((i as u64) * (interval_ns as u64)),
                    time_elapsed_ms: 0.0,
                    pipeline_id: 0,
                });
        }

        let forecast_fn = basic_probability_forecast::History::<
            basic_probability_forecast::BasicCategory,
        >::forecast_fn();
        let known_bins = vec![time_series::HARD_CLASS_BIN; 16]; // Batch size 16

        // Scenario: 1000ms window, 10 item cap
        // Expected items in window: 1000ms / 1ms = 1000 items.
        // Cap: 10 items.
        // ratio_kept should be (16 / 1000) = 0.016 if we return 16 items.
        // Current bug returns 1.0 because 16 >= 10.

        let future_kind = FutureWindowKind::TimeWithMaximumCount {
            time_ms: 1000,
            max_count: 10,
        };

        let (forecast, ratio_kept) = forecast_fn(&mut history, &known_bins, future_kind);

        assert_eq!(forecast.len(), 16, "Should return all known bins");
        assert!(
            ratio_kept < 0.1,
            "Ratio kept should be normalized. Got: {}",
            ratio_kept
        );
    }
}

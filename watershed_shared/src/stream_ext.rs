use std::{task::Poll, time::Duration};

use futures::{future::BoxFuture, stream::{Stream, StreamExt}, FutureExt};
use log::{debug, warn};
// use tokio::sync::watch;

use crate::{basic_pooling::get_tuple_vec, Tuple};


pub trait StreamExtPlus: Stream {
    // use a stream of futures to tell when to buffer a different stream's items
    // fn chunk_until<S, T>(self, until: S) -> ChunkUntil<Self, S>
    // where
    //     Self: Sized,
    //     S: Stream<Item=watch::Receiver<T>>,
    // {
    //     ChunkUntil::new(self, until)
    // }
    // fn chunk_n_tuples(self, n: usize) -> ChunkNTuples<Self> where Self: Sized {
    //     ChunkNTuples::new(self, n)
    // }
    // fn chunk_tuples_until<T>(self, n: usize, until: watch::Receiver<T>) -> ChunkTuplesUntil<Self, T> where Self: Sized {
    //     ChunkTuplesUntil::new(self, n, until)
    // }
    // fn chunk_tuples<T>(self, max_count: usize, until: Receiver<T>) -> ChunkTuples<Self> where Self: Sized {
    //     ChunkTuples::new(self, max_count, until)
    // }
    fn chunk_tuples_timeout(
        self,
        max_count: usize,
        max_time: Duration,
    ) -> ChunkTuplesTimeout<Self>
    where
        Self: Sized + Unpin + Stream<Item = Tuple>,
    {
        ChunkTuplesTimeout::new(self, max_count, max_time)
    }
}

impl<S> StreamExtPlus for S where S: Stream {}

// pub struct ChunkUntil<S1, S2>
// where
//     S1: Stream,
// {
//     stream: S1,
//     chunk_state: Vec<S1::Item>,
//     until_stream: S2,
// }

// impl<S1, S2, T> ChunkUntil<S, T>
// where
//     S: Stream,
//     S2: Stream<Item=watch::Receiver<T>>,
// {
//     fn new(stream: S, until: watch::Receiver<T>) -> Self {
//         ChunkUntil {
//             stream,
//             until,
//             chunk_state: Vec::new(),
//         }
//     }
// }

// impl<S, T> Stream for ChunkUntil<S, T>
// where
//     S: Stream,

// {
//     type Item = S::Item;

//     fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
//         todo!()
//     }
// }

pub struct ChunkTuplesTimeout<S>
where
    S: Stream<Item = Tuple>,
{
    stream: S,
    max_count: usize,
    current_chunk_state: Vec<S::Item>,
    current_chunk_start_time: std::time::Instant,
    max_time: Duration,
    current_timeout: BoxFuture<'static, ()>,
}

impl<S> ChunkTuplesTimeout<S>
where
    S: Stream<Item = Tuple>,
{
    fn new(stream: S, max_count: usize, max_time: Duration) -> Self {
        ChunkTuplesTimeout {
            stream,
            max_count,
            max_time,
            current_chunk_state: get_tuple_vec(),
            current_chunk_start_time: std::time::Instant::now(),
            current_timeout: tokio::time::timeout(
                max_time,
                std::future::pending::<()>(),
            ).map(|v| v.unwrap_or_default()).boxed(),
        }
    }
    fn reset_timer(&mut self) {
        self.current_chunk_start_time = std::time::Instant::now();
        self.current_timeout = tokio::time::timeout(
            self.max_time,
            std::future::pending::<()>(),
        ).map(|v| v.unwrap_or_default()).boxed();
    }
}

impl<S> Stream for ChunkTuplesTimeout<S>
where
    S: Stream<Item = Tuple> + Unpin,
{
    type Item = Vec<S::Item>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        debug!("polling chunked stream");
        loop {
            debug!("start of chunked stream poll loop");
            let is_elapsed_future = self.current_timeout.as_mut().poll_unpin(cx);
            if let Poll::Ready(_) = is_elapsed_future {
                debug!("time has elapsed. time to output the vec");
                self.reset_timer();
                if self.current_chunk_state.len() == 0 {
                    warn!("time has elapsed but we have no items to output");
                    continue;
                    // return Poll::Pending;
                }
                debug!("time has elapsed after checking. time to output {} items", self.current_chunk_state.len());
                let output_vec =
                    std::mem::replace(&mut self.current_chunk_state, get_tuple_vec());
                return Poll::Ready(Some(output_vec));
            }
            debug!("time has not elapsed yet so we are polling");
            let next_item_future = self.stream.poll_next_unpin(cx);
            let Poll::Ready(next_item) = next_item_future else {
                debug!("underlying stream is not ready yet after polling so we will wait");
                // if self.current_chunk_start_time.elapsed() >= self.max_time {
                //     self.reset_timer();
                //     if self.current_chunk_state.len() == 0 {
                //         return Poll::Pending;
                //     }
                //     debug!("time has elapsed after checking. time to output the vec");
                //     let output_vec =
                //         std::mem::replace(&mut self.current_chunk_state, get_tuple_vec());
                //     return Poll::Ready(Some(output_vec));
                // }
                // we aren't full or timed out yet, but we're out of items for now
                return Poll::Pending;
            };
            match next_item {
                // inner stream had new items
                Some(v) => {
                    debug!("inner stream had a new item");
                    self.current_chunk_state.push(v);
                    if self.current_chunk_state.len() >= self.max_count
                        // || self.current_chunk_start_time.elapsed() >= self.max_time
                    {
                        let output_vec =
                            std::mem::replace(&mut self.current_chunk_state, get_tuple_vec());
                        self.reset_timer();
                        return Poll::Ready(Some(output_vec));
                    }
                }
                // inner stream is done
                None => {
                    debug!("inner stream is done."); 
                    let output_vec =
                        std::mem::replace(&mut self.current_chunk_state, get_tuple_vec());
                    // make sure it always gets to poll the stream again to see that it's empty
                    self.current_timeout = std::future::pending::<()>().boxed();
                    if output_vec.len() > 0 {
                        debug!("inner stream is done and we had something to output"); 
                        return Poll::Ready(Some(output_vec));
                    } else {
                        debug!("inner stream is done and we had nothing left to output"); 
                        return Poll::Ready(None);
                    }
                }
            }
        }
    }
}



#[cfg(test)]
mod tests {
    use futures::stream;
    use tap::Tap;
    use tokio::time::sleep;

    use super::*;

    #[tokio::test]
    async fn test_chunk_tuples_timeout_max_count() {
        let input_stream = stream::iter(vec![
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 1i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 2i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 3i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 4i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 5i32.into());
            }),
        ]);

        let chunked_stream = ChunkTuplesTimeout::new(input_stream, 3, Duration::from_secs(10));
        let collected_chunks: Vec<Vec<Tuple>> = chunked_stream.collect().await;

        assert_eq!(collected_chunks.len(), 2);
        assert_eq!(
            collected_chunks[0],
            vec![
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 1i32.into());
                }),
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 2i32.into());
                }),
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 3i32.into());
                }),
            ]
        );
        assert_eq!(
            collected_chunks[1],
            vec![
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 4i32.into());
                }),
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 5i32.into());
                }),
            ]
        );
    }

    #[tokio::test]
    async fn test_chunk_tuples_timeout_max_time() {
        let input_stream = stream::iter(vec![
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 1i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 2i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 3i32.into());
            }),
        ]);
        // we follow it up with a pending stream because we want to test the timeout
        println!("g0");
        let input_stream = input_stream.chain(stream::pending());
        println!("g1");

        let starting_time = std::time::Instant::now();
        println!("g2");
        let mut chunked_stream = ChunkTuplesTimeout::new(input_stream, 10, Duration::from_millis(100));
        println!("g3");
        let collected_chunks: Option<Vec<Tuple>> = chunked_stream.next().await;
        println!("g4");
        assert!(collected_chunks.is_some());
        println!("g5");
        let collected_chunks = collected_chunks.unwrap();
        let elapsed_time = starting_time.elapsed();

        assert_eq!(collected_chunks.len(), 3);
        assert_eq!(
            collected_chunks,
            vec![
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 1i32.into());
                }),
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 2i32.into());
                }),
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 3i32.into());
                }),
            ]
        );
        assert!(elapsed_time >= Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_chunk_tuples_timeout_mixed() {
        let input_stream = stream::iter(vec![
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 1i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 2i32.into());
            }),
        ]);

        let chunked_stream = ChunkTuplesTimeout::new(input_stream, 2, Duration::from_millis(50));
        let collected_chunks: Vec<Vec<Tuple>> = chunked_stream.collect().await;

        assert_eq!(collected_chunks.len(), 1);
        assert_eq!(
            collected_chunks[0],
            vec![
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 1i32.into());
                }),
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 2i32.into());
                }),
            ]
        );
    }

    #[tokio::test]
    async fn test_chunk_tuples_timeout_empty() {
        let input_stream = stream::iter(Vec::<Tuple>::new());

        let chunked_stream = ChunkTuplesTimeout::new(input_stream, 3, Duration::from_secs(1));
        let collected_chunks: Vec<Vec<Tuple>> = chunked_stream.collect().await;

        assert_eq!(collected_chunks.len(), 0);
    }

    #[tokio::test]
    async fn test_chunk_tuples_timeout_partial_chunk() {
        let input_stream = stream::iter(vec![
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 1i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 2i32.into());
            }),
            Tuple::new().tap_mut(|t| {
                t.insert("a".into(), 3i32.into());
            }),
        ]);

        let chunked_stream = ChunkTuplesTimeout::new(input_stream, 5, Duration::from_millis(50));
        let collected_chunks: Vec<Vec<Tuple>> = chunked_stream.collect().await;

        assert_eq!(collected_chunks.len(), 1);
        assert_eq!(
            collected_chunks[0],
            vec![
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 1i32.into());
                }),
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 2i32.into());
                }),
                Tuple::new().tap_mut(|t| {
                    t.insert("a".into(), 3i32.into());
                }),
            ]
        );
    }
}


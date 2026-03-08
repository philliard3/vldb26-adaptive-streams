use core::f32;
use std::any::Any;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use image::GenericImageView;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use watershed_shared::basic_pooling::{get_tuple, get_tuple_vec, return_tuple};
use watershed_shared::caching::StrToKey;
use watershed_shared::global_logger::{LimitedHabValue, NO_AUX_DATA};
use watershed_shared::operators::AggregationResult;
use watershed_shared::{HabString, HabValue, Tuple};

use serde::{Deserialize, Serialize};
use serde_bytes;

use watershed_shared::devec::DeVec as Queue;

use crate::omz_utils;

// const FACE_COUNT_FIELD: &str = "total_face_count";
pub const FACE_COUNT_FIELD: &str = "boxes_detected";
pub const INDIVIDUAL_BOX_BOUND_FIELD: &str = "my_bounding_box";
pub const INDIVIDUAL_BOX_ID_FIELD: &str = "my_box_number";
pub const ORIGINAL_IMAGE_ID_FIELD: &str = "img_id";
pub const ORIGINAL_IMAGE_ID_INT_FIELD: &str = "img_id_int";
pub const FACES_DETECTED_FIELD: &str = FACE_COUNT_FIELD;
// const ORIGINAL_IMAGE_FIELD: &str = "original_img";
pub const ORIGINAL_IMAGE_FIELD: &str = "image";
// const ORIGINAL_IMAGE_SHAPE_FIELD: &str = "original_img_shape";
pub const ORIGINAL_IMAGE_SHAPE_FIELD: &str = "shape";
pub const BLURRED_IMAGE_FIELD: &str = "blurred_img";
pub const EXPECTED_MATCHES_FIELD: &str = "expected_matches";

fn gaussian_blur_regions(
    orig_img: &image::RgbImage,
    regions: &[(u32, u32, u32, u32)],
    factor: f32,
) -> image::RgbImage {
    let mut orig_img = orig_img.clone();
    for region in regions {
        let (x, y, w, h) = *region;
        // let region = imageproc::rect::Rect::at(x as i32, y as i32).of_size(w as u32, h as u32);
        let region = orig_img.view(x as u32, y as u32, w as u32, h as u32);
        // imageproc::filter::gaussian_blur(&mut orig_img, factor, region);
        let blurred_region = image::imageops::blur(&region.to_image(), factor);
        // copy it back to the relevant area of orig_img
        for (x, y, pixel) in blurred_region.enumerate_pixels() {
            orig_img.put_pixel(x, y, *pixel);
        }
    }
    let blurred_img = orig_img;
    blurred_img
}

#[derive(Debug, Serialize, Deserialize)]
struct DetectFacesInput<'a> {
    tuple_id: u64,
    dims: Vec<u32>,
    #[serde(with = "serde_bytes")]
    img_buf: &'a [u8],
}
const IMAGE_FIELD_NAME: &str = "image";
const RGB_NUM_CHANNELS: usize = 3;
pub(crate) fn encode_image_for_detection(tuple_id: usize, tuple: &Tuple) -> zeromq::ZmqMessage {
    let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        error!("system time before unix epoch");
        return shutdown_sequence_detect();
    };
    let now = now.as_nanos();
    let diff = now - tuple.unix_time_created_ns();
    trace!("encoding sequence for tuple {tuple_id} with time difference {diff} ns");
    let Some(byte_buffer) = tuple.get(IMAGE_FIELD_NAME) else {
        error!(
            "image field not found in tuple {tuple_id}. available fields are {:?}",
            tuple.keys().collect::<Vec<_>>()
        );
        return shutdown_sequence_detect();
    };
    let Some(byte_buffer) = byte_buffer.as_byte_buffer() else {
        error!("image field not a byte buffer in tuple {tuple_id}");
        return shutdown_sequence_detect();
    };

    let byte_buffer = byte_buffer.as_ref();
    let Some(original_width) = tuple.get("original_width") else {
        error!(
            "original width field not found in tuple {tuple_id}. available fields are {:?}",
            tuple.keys().collect::<Vec<_>>()
        );
        return shutdown_sequence_detect();
    };
    let Some(original_width) = original_width.as_integer() else {
        error!("original width field not an integer in tuple {tuple_id}");
        return shutdown_sequence_detect();
    };
    let Some(original_height) = tuple.get("original_height") else {
        error!(
            "original height field not found in tuple {tuple_id}. available fields are {:?}",
            tuple.keys().collect::<Vec<_>>()
        );
        return shutdown_sequence_detect();
    };
    let Some(original_height) = original_height.as_integer() else {
        error!("original height field not an integer in tuple {tuple_id}");
        return shutdown_sequence_detect();
    };
    let dims = vec![
        original_height as u32,
        original_width as u32,
        RGB_NUM_CHANNELS as u32,
    ];
    let tensor_message = DetectFacesInput {
        tuple_id: tuple_id as u64,
        dims,
        img_buf: byte_buffer,
    };
    zeromq::ZmqMessage::from(rmp_serde::to_vec(&tensor_message).unwrap_or_else(|e| {
        error!("failed to serialize tensor message: {e}");
        vec![]
    }))
}

#[derive(Debug, Serialize, Deserialize)]
struct DetectFacesOutput {
    tuple_id: usize,
    boxes_detected: u32,
    #[serde(with = "serde_bytes")]
    boxes_tensor: Vec<u8>,
}

pub(crate) fn decode_bounding_boxes_from_detection(
    msg: zeromq::ZmqMessage,
    tuple_map: &DashMap<usize, Tuple>,
) -> Vec<Tuple> {
    let Ok(msg) = rmp_serde::from_read::<_, DetectFacesOutput>(watershed_shared::FrameReader::new(
        msg.into_vec(),
    )) else {
        error!("failed to deserialize bounding boxes");
        return vec![];
    };
    let Some((_tuple_id, mut tuple)) = tuple_map.remove(&msg.tuple_id) else {
        error!(
            "failed to find tuple with id {} for bounding box decoding",
            msg.tuple_id
        );
        return vec![];
    };
    //  recast as ndarray
    let boxes_detected = msg.boxes_detected as usize;
    if boxes_detected == 0 {
        // per meeting on 2025-04-11, we just drop it if there's nothing detectd
        return_tuple(tuple);
        vec![]
    } else {
        // when there's at least one, we make a field in the tuple for it and just forward it along.
        // we will make a my_bounding_box field later after embedding

        let mut tuples = get_tuple_vec();
        // add the last one
        tuple.insert(
            FACE_COUNT_FIELD.into(),
            HabValue::Integer(boxes_detected as _),
        );
        // bounding boxes field will be split up later
        tuple.insert(
            "bounding_boxes".into(),
            HabValue::ByteBuffer(msg.boxes_tensor),
        );
        debug!(
            "decoded {} bounding boxes for tuple {}",
            boxes_detected, msg.tuple_id
        );
        tuples.push(tuple);
        tuples
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingInput<'a> {
    tuple_id: u64,
    img_dims: Vec<u32>,
    #[serde(with = "serde_bytes")]
    img_buf: &'a [u8],
    num_boxes: u32,
    #[serde(with = "serde_bytes")]
    box_buffer: &'a [u8],
}

pub(crate) fn encode_for_embedding(tuple_id: usize, tuple: &Tuple) -> zeromq::ZmqMessage {
    debug!(
        "encoding tuple with id= {tuple_id} and image id={:?} for embedding",
        tuple.get(ORIGINAL_IMAGE_ID_FIELD)
    );
    let Some(byte_buffer) = tuple.get(IMAGE_FIELD_NAME) else {
        error!(
            "image field not found in tuple {tuple_id}. available fields are {:?}",
            tuple.keys().collect::<Vec<_>>()
        );
        return shutdown_sequence_embed();
    };
    let Some(byte_buffer) = byte_buffer.as_byte_buffer() else {
        error!("image field not a byte buffer in tuple {tuple_id}");
        return shutdown_sequence_embed();
    };
    let byte_buffer = byte_buffer.as_ref();

    let Some(original_width) = tuple.get("original_width") else {
        error!(
            "original width field not found in tuple {tuple_id}. available fields are {:?}",
            tuple.keys().collect::<Vec<_>>()
        );
        return shutdown_sequence_embed();
    };
    let Some(original_width) = original_width.as_integer() else {
        error!("original width field not an integer in tuple {tuple_id}");
        return shutdown_sequence_embed();
    };

    let Some(original_height) = tuple.get("original_height") else {
        error!("original height field not found");
        return shutdown_sequence_embed();
    };
    let Some(original_height) = original_height.as_integer() else {
        error!("original height field not an integer");
        return shutdown_sequence_embed();
    };

    let Some(boxes_detected) = tuple.get(FACE_COUNT_FIELD) else {
        error!("boxes detected field not found");
        return shutdown_sequence_embed();
    };
    let Some(boxes_detected) = boxes_detected.as_integer() else {
        error!("boxes detected field not an integer");
        return shutdown_sequence_embed();
    };

    let dims = vec![
        original_height as u32,
        original_width as u32,
        RGB_NUM_CHANNELS as u32,
    ];
    let Some(bounding_boxes) = tuple.get("bounding_boxes") else {
        error!(
            "bounding boxes field not found in tuple {tuple_id}. available fields are {:?}",
            tuple.keys().collect::<Vec<_>>()
        );
        return shutdown_sequence_embed();
    };
    let Some(bounding_boxes) = bounding_boxes.as_byte_buffer() else {
        error!("bounding boxes field not a byte buffer in tuple {tuple_id}");
        return shutdown_sequence_embed();
    };
    let bounding_boxes = bounding_boxes.as_ref();
    let tensor_message = EmbeddingInput {
        tuple_id: tuple_id as u64,
        img_dims: dims,
        img_buf: byte_buffer,
        num_boxes: boxes_detected as u32,
        // if nothing was detected, we expect to receive nothing back
        box_buffer: bounding_boxes,
    };
    zeromq::ZmqMessage::from(rmp_serde::to_vec(&tensor_message).unwrap_or_else(|e| {
        error!("failed to serialize tensor for embedding: {e}");
        vec![]
    }))
}

pub(crate) fn shutdown_sequence_detect() -> zeromq::ZmqMessage {
    zeromq::ZmqMessage::from(
        rmp_serde::to_vec(&DetectFacesInput {
            tuple_id: u64::MAX,
            dims: Default::default(),
            img_buf: &[],
        })
        .unwrap_or_else(|e| {
            let msg = format!("failed to serialize shutdown message for python detection udf");
            error!("{msg}: {e}");
            panic!("{msg}: {e}");
        }),
    )
}

pub(crate) fn shutdown_sequence_embed() -> zeromq::ZmqMessage {
    zeromq::ZmqMessage::from(
        rmp_serde::to_vec(&EmbeddingInput {
            tuple_id: u64::MAX,
            img_dims: Default::default(),
            img_buf: &[],
            num_boxes: 0,
            box_buffer: &[],
        })
        .unwrap_or_else(|e| {
            // "failed to serialize shutdown message for python embedding udf"),
            let msg = format!("failed to serialize shutdown message for python embedding udf");
            error!("{msg}: {e}");
            panic!("{msg}: {e}");
        }),
    )
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingMessage {
    tuple_id: u64,
    // dims: Vec<u64>,
    dims: Vec<u32>,
    #[serde(with = "serde_bytes")]
    tensor: Vec<u8>,
}

pub(crate) fn decode_embedding(
    msg: zeromq::ZmqMessage,
    tuple_map: &DashMap<usize, Tuple>,
) -> Vec<Tuple> {
    let Ok(msg) = rmp_serde::from_read::<_, EmbeddingMessage>(watershed_shared::FrameReader::new(
        msg.into_vec(),
    )) else {
        error!("failed to deserialize embedding");
        return vec![];
    };
    let Some((_tuple_id, mut tuple)) = tuple_map.remove(&(msg.tuple_id as _)) else {
        error!(
            "failed to find tuple with id {} for embedding decoding",
            msg.tuple_id
        );
        return vec![];
    };
    let byte_buffer: Vec<u8> = msg.tensor;
    const EXPECTED_EMBEDDING_LENGTH: usize = 512;
    // check if the shape is correct
    if msg.dims.len() != 2 {
        error!(
            "embedding shape mismatch: expected (n x {:?}), got {:?}",
            EXPECTED_EMBEDDING_LENGTH, msg.dims
        );
        return vec![];
    }
    if EXPECTED_EMBEDDING_LENGTH != msg.dims[1] as usize {
        error!(
            "embedding length mismatch: expected {:?}, got {:?}",
            EXPECTED_EMBEDDING_LENGTH, msg.dims[1]
        );
        return vec![];
    }
    // put into tuple
    // let shape = msg.dims;
    // let shape = shape.iter().map(|x| *x as usize).collect::<Vec<_>>();
    // tuple.insert("embedding".into(), HabValue::ByteBuffer(byte_buffer));
    let float_stride = std::mem::size_of::<f32>();
    let num_embeddings = msg.dims[0] as usize;
    let embedding_stride = msg.dims[1] as usize;

    let mut embedding_buffer_iter = byte_buffer.chunks_exact(embedding_stride * float_stride);
    let bounding_box_buffer = tuple.get("bounding_boxes").unwrap_or_else(|| {
        error!("bounding boxes field not found");
        static NULL: HabValue = HabValue::Null;
        &NULL
    });
    let Some(bounding_box_buffer) = bounding_box_buffer.as_byte_buffer() else {
        error!("bounding boxes field not a byte buffer");
        return vec![];
    };
    let mut bounding_box_buffer_iter = bounding_box_buffer
        .as_ref()
        .chunks_exact(4 * std::mem::size_of::<f32>())
        .map(|x| {
            let bbox = x
                .chunks_exact(4)
                .map(|x| f32::from_ne_bytes(x.try_into().unwrap()) as i32)
                .collect::<Vec<_>>();
            HabValue::IntBuffer(bbox)
        });

    // multiple embeddings must be split up to use for the chroma lookup
    let mut tuples = get_tuple_vec();
    tuples.reserve_exact(num_embeddings);
    for box_number in 0..(num_embeddings - 1) {
        let Some(chunk) = embedding_buffer_iter.next() else {
            error!("no more embeddings left!");
            return vec![];
        };
        let embedding = chunk
            .chunks_exact(float_stride)
            .map(|x| HabValue::Float((f32::from_ne_bytes(x.try_into().unwrap()) as f64).into()))
            .collect::<Vec<_>>();
        let mut new_tuple = get_tuple();
        new_tuple.insert("embedding".into(), HabValue::List(embedding));
        for (key, value) in tuple.iter() {
            if key != IMAGE_FIELD_NAME && key != watershed_shared::basic_pooling::UUID_FIELD {
                new_tuple.insert(key.clone(), value.clone());
            }
        }
        let my_bounding_box = bounding_box_buffer_iter.next().unwrap_or_else(|| {
            error!("bounding box field not found");
            HabValue::Null
        });
        new_tuple.insert(INDIVIDUAL_BOX_BOUND_FIELD.into(), my_bounding_box);
        new_tuple.insert(
            INDIVIDUAL_BOX_ID_FIELD.into(),
            HabValue::Integer(box_number as i32),
        );
        tuples.push(new_tuple);
    }
    // add the last one
    let Some(chunk) = embedding_buffer_iter.next() else {
        error!("no more embeddings left!");
        return vec![];
    };
    let my_bounding_box = bounding_box_buffer_iter.next().unwrap_or_else(|| {
        error!("bounding box field not found");
        HabValue::Null
    });
    drop(bounding_box_buffer);
    tuple.insert(INDIVIDUAL_BOX_BOUND_FIELD.into(), my_bounding_box);
    tuple.insert(
        INDIVIDUAL_BOX_ID_FIELD.into(),
        HabValue::Integer((num_embeddings - 1) as i32),
    );
    let embedding = chunk
        .chunks_exact(float_stride)
        .map(|x| HabValue::Float((f32::from_ne_bytes(x.try_into().unwrap()) as f64).into()))
        .collect::<Vec<_>>();
    tuple.insert("embedding".into(), HabValue::List(embedding));
    tuples.push(tuple);

    'log_decode_embedding: {
        use watershed_shared::global_logger;
        let log_location = "decode_embedding".to_raw_key();
        let aux_data = NO_AUX_DATA;
        debug!(
            "decoded {} embeddings for tuple {}",
            tuples.len(),
            msg.tuple_id
        );
        if let Err(e) = global_logger::log_data(msg.tuple_id as _, log_location, aux_data) {
            for err in e {
                error!("failed to log decode embedding: {err}");
            }
            break 'log_decode_embedding;
        }
    }

    tuples
}

pub(crate) fn extract_image_from_tuple(
    t: &mut Tuple,
    shape_field_name: HabString,
    buffer_field_name: HabString,
) -> image::RgbImage {
    let Some(img) = t.get(&buffer_field_name) else {
        error!(
            "image field not found in tuple {:?} with img_id={:?}. available fields are {:?}",
            t.id(),
            t.get(ORIGINAL_IMAGE_ID_FIELD),
            t.keys().collect::<Vec<_>>()
        );
        panic!(
            "image field not found in tuple {:?} with img_id={:?}. available fields are {:?}",
            t.id(),
            t.get(ORIGINAL_IMAGE_ID_FIELD),
            t.keys().collect::<Vec<_>>()
        );
    };

    let img = match img {
        HabValue::ByteBuffer(_) => img.into_byte_buffer().unwrap_or_else(|| {
            panic!(
                "we just checked it was a byte buffer in tuple {:?} with img_id={:?}",
                t.id(),
                t.get(ORIGINAL_IMAGE_ID_FIELD)
            )
        }),
        HabValue::SharedArrayU8(ref b) => b.0.iter().cloned().collect::<Vec<_>>(),
        _ => {
            error!(
                "image field not a byte buffer or shared array in tuple {:?} with img_id={:?}",
                t.id(),
                t.get(ORIGINAL_IMAGE_ID_FIELD)
            );
            panic!(
                "image field not a byte buffer or shared array in tuple {:?} with img_id={:?}",
                t.id(),
                t.get(ORIGINAL_IMAGE_ID_FIELD)
            );
        }
    };
    let Some(shape) = t.get(&shape_field_name) else {
        error!("shape field {shape_field_name} not found in tuple {:?} with img_id={:?}. available fields are {:?}", t.id(), t.get(ORIGINAL_IMAGE_ID_FIELD), t.keys().collect::<Vec<_>>());
        panic!("shape field {shape_field_name} not found in tuple {:?} with img_id={:?}. available fields are {:?}", t.id(), t.get(ORIGINAL_IMAGE_ID_FIELD), t.keys().collect::<Vec<_>>());
    };
    let Some(shape) = shape.as_shape_buffer() else {
        error!(
            "shape field {shape_field_name} not a shape buffer in tuple {:?} with img_id={:?}",
            t.id(),
            t.get(ORIGINAL_IMAGE_ID_FIELD)
        );
        panic!(
            "shape field {shape_field_name} not a shape buffer in tuple {:?} with img_id={:?}",
            t.id(),
            t.get(ORIGINAL_IMAGE_ID_FIELD)
        );
    };
    let img_len = img.len();
    let Some(img) = image::RgbImage::from_raw(shape[0] as u32, shape[1] as u32, img) else {
        let msg = format!(
            "failed to create image from raw data in tuple {:?} with img_id={:?}. shape = {:?}, buffer length = {}",
            t.id(),
            t.get(ORIGINAL_IMAGE_ID_FIELD),
            shape,
            img_len
        );
        error!("{msg}");
        panic!("{msg}");
    };
    // DynamicImage::ImageRgb8(img)
    img
}

pub(crate) fn insert_image_into_tuple(
    t: &mut Tuple,
    shape_field_name: HabString,
    buffer_field_name: HabString,
    final_image: image::RgbImage,
) {
    t.insert(
        shape_field_name,
        HabValue::ShapeBuffer(vec![
            final_image.height() as usize,
            final_image.width() as usize,
            3,
        ]),
    );
    let img = final_image.into_raw();
    t.insert(buffer_field_name, HabValue::ByteBuffer(img));
}

pub(crate) fn global_log_did_blur(
    tuple_id: usize,
    image_id: &str,
    box_number: usize,
    blurred: bool,
) {
    use watershed_shared::global_logger;
    let log_location = "did_blur".to_raw_key();
    let aux_data = Some(HashMap::from([
        (
            "image_id".to_raw_key(),
            LimitedHabValue::String(image_id.to_key()),
        ),
        (
            "box_number".to_raw_key(),
            LimitedHabValue::Integer(box_number as _),
        ),
        (
            "blurred".to_raw_key(),
            LimitedHabValue::Integer(blurred as _),
        ),
    ]));
    if let Err(e) = global_logger::log_data(tuple_id as _, log_location, aux_data) {
        for err in e {
            error!("failed to log blur: {err}");
        }
    }
}

pub(crate) fn get_blur_regions<'a>(
    tuples: impl 'a + IntoIterator<Item = &'a Tuple>,
    region_field_name: HabString,
    should_blur_predicate: fn(&Tuple) -> bool,
) -> impl 'a + Iterator<Item = (u32, u32, u32, u32)> {
    tuples.into_iter().filter_map(move |tuple| {
        let Some(box_number) = tuple.get(INDIVIDUAL_BOX_ID_FIELD) else {
            error!("my_box_number field not found");
            return None;
        };
        let Some(box_number) = box_number.as_integer() else {
            error!("my_box_number field not an integer");
            return None;
        };
        let Some(image_id) = tuple.get(ORIGINAL_IMAGE_ID_FIELD) else {
            error!("original image id field not found");
            return None;
        };
        let Some(image_id) = image_id.as_string() else {
            error!("original image id field not a string");
            return None;
        };
        let should_blur = should_blur_predicate(tuple);
        global_log_did_blur(
            tuple.id() as _,
            image_id.as_str(),
            box_number as _,
            should_blur,
        );
        if !should_blur {
            debug!("skipping tuple {:?}'s face (image id = {:?} ) for blurring", tuple.id(), image_id);
            return None;
        }

        let Some(region) = tuple.get(&region_field_name) else {
            error!("region field not found");
            return None;
        };
        let Some(region) = region.as_int_buffer() else {
            error!("region field not an int buffer");
            return None;
        };
        if region.len() != 4 {
            return None;
        }
        let width = region[2] - region[0];
        let height = region[3] - region[1];
        if width <= 0 || height <= 0 {
            error!(
                "skipping tuple {:?}'s face ( image id = {:?} ) for blurring because it has no area: width={} height={}",
                tuple.id(),
                image_id,
                width,
                height
            );
            return None;
        }

        if width <= 5 || height <= 5 {
            warn!(
                "skipping tuple {:?}'s face for blurring because it has very small area: width={} height={}",
                tuple.id()
                , width, height
            );
            return None;
        }
        debug!("blurring tuple {:?}'s face", tuple.id());

        Some((
            region[0] as u32,
            region[1] as u32,
            width as u32,
            height as u32,
        ))
    })
}

const DEFAULT_BLUR_COUNT: usize = 16;
pub(crate) fn blur_regions(
    img: &mut image::RgbImage,
    regions: impl Iterator<Item = (u32, u32, u32, u32)>,
) {
    let overall_start = std::time::Instant::now();
    let mut stats = smallvec::SmallVec::<[(f64, f64); DEFAULT_BLUR_COUNT]>::new();
    for (x, y, w, h) in regions {
        let region = img.view(
            x as u32,
            y as u32,
            (w as u32).min(img.width().saturating_sub(x as u32).saturating_sub(1)),
            (h as u32).min(img.height().saturating_sub(y as u32).saturating_sub(1)),
        );
        let blur_start = std::time::Instant::now();
        let blurred_region = image::imageops::fast_blur(&region.to_image(), 5.0);
        let blur_elapsed_nanos = blur_start.elapsed().as_nanos();
        // copy it back to the relevant area of orig_img
        let transfer_start = std::time::Instant::now();
        for (inner_x, inner_y, pixel) in blurred_region.enumerate_pixels() {
            img.put_pixel(inner_x + x, inner_y + y, *pixel);
        }
        let transfer_elapsed_nanos = transfer_start.elapsed().as_nanos();
        stats.push((blur_elapsed_nanos as _, transfer_elapsed_nanos as _));
    }
    if stats.is_empty() {
        warn!("no regions to blur");
        return;
    }
    let overall_elapsed_nanos = overall_start.elapsed().as_nanos();
    let stat_len = stats.len();
    let mean_blur_time = stats.iter().map(|(blur, _)| *blur).sum::<f64>() / stat_len as f64;
    let mean_transfer_time =
        stats.iter().map(|(_, transfer)| *transfer).sum::<f64>() / stat_len as f64;
    debug!(
        "blurred {} regions in {} nanos, mean blur time: {:.2} nanos, mean transfer time: {:.2} nanos",
        stat_len,
        overall_elapsed_nanos,
        mean_blur_time,
        mean_transfer_time
    );
}

//what about a box blur to compare against?
fn box_blur_regions<
    const KERNEL_SIZE: usize,
    const STRIDE_SIZE: usize,
    const INTERNAL_STRIDE_SIZE: usize,
>(
    img: &mut image::RgbImage,
    regions: impl Iterator<Item = (u32, u32, u32, u32)>,
) {
    let kernel_size: usize = KERNEL_SIZE;
    let stride_size: usize = STRIDE_SIZE;
    let inner_stride_size: usize = INTERNAL_STRIDE_SIZE;
    let overall_start = std::time::Instant::now();
    let mut stats = smallvec::SmallVec::<[(f64, f64); DEFAULT_BLUR_COUNT]>::new();
    for (x, y, w, h) in regions {
        let region = img.view(
            x,
            y,
            w.min(img.width().saturating_sub(x).saturating_sub(1)),
            h.min(img.height().saturating_sub(y).saturating_sub(1)),
        );
        let blur_start = std::time::Instant::now();
        let mut blurred_region_image = region.to_image();
        // there is no box blur in the imageops crate, so we have to implement it ourselves
        for y in (0..blurred_region_image.height()).step_by(stride_size as _) {
            for x in (0..blurred_region_image.width()).step_by(stride_size as _) {
                let mut r_sum = 0f64;
                let mut g_sum = 0f64;
                let mut b_sum = 0f64;
                let mut count = 0f64;
                let mut inner_stride_count = 0;
                for ky in -(kernel_size as i32 / 2)..=(kernel_size as i32 / 2) {
                    for kx in -(kernel_size as i32 / 2)..=(kernel_size as i32 / 2) {
                        let nx = x as i32 + kx;
                        let ny = y as i32 + ky;
                        if nx >= 0
                            && ny >= 0
                            && nx < blurred_region_image.width() as i32
                            && ny < blurred_region_image.height() as i32
                        {
                            inner_stride_count += 1;
                            if inner_stride_count as usize % inner_stride_size != 0 {
                                continue;
                            }
                            let pixel = blurred_region_image.get_pixel(nx as u32, ny as u32);
                            r_sum += pixel[0] as f64;
                            g_sum += pixel[1] as f64;
                            b_sum += pixel[2] as f64;
                            count += 1.0;
                        }
                    }
                }
                if count > 0.0 {
                    let r = (r_sum / count) as u8;
                    let g = (g_sum / count) as u8;
                    let b = (b_sum / count) as u8;
                    // blurred_region_image.put_pixel(x, y, image::Rgb([r, g, b]));
                    // assign to all pixels in the kernel area
                    for ky in -(kernel_size as i32 / 2)..=(kernel_size as i32 / 2) {
                        for kx in -(kernel_size as i32 / 2)..=(kernel_size as i32 / 2) {
                            let nx = x as i32 + kx;
                            let ny = y as i32 + ky;
                            if nx >= 0
                                && ny >= 0
                                && nx < blurred_region_image.width() as i32
                                && ny < blurred_region_image.height() as i32
                            {
                                blurred_region_image.put_pixel(
                                    nx as u32,
                                    ny as u32,
                                    image::Rgb([r, g, b]),
                                );
                            }
                        }
                    }
                }
            }
        }
        let blur_elapsed_nanos = blur_start.elapsed().as_nanos();
        // copy it back to the relevant area of orig_img
        let transfer_start = std::time::Instant::now();
        for (inner_x, inner_y, pixel) in blurred_region_image.enumerate_pixels() {
            img.put_pixel(inner_x + x, inner_y + y, *pixel);
        }
        let transfer_elapsed_nanos = transfer_start.elapsed().as_nanos();
        stats.push((blur_elapsed_nanos as _, transfer_elapsed_nanos as _));
    }
    if stats.is_empty() {
        warn!("no regions to blur");
        return;
    }
    let overall_elapsed_nanos = overall_start.elapsed().as_nanos();
    let stat_len = stats.len();
    let mean_blur_time = stats.iter().map(|(blur, _)| *blur).sum::<f64>() / stat_len as f64;
    let mean_transfer_time =
        stats.iter().map(|(_, transfer)| *transfer).sum::<f64>() / stat_len as f64;
    debug!(
        "blurred {} regions in {} nanos, mean blur time: {:.2} nanos, mean transfer time: {:.2} nanos",
        stat_len,
        overall_elapsed_nanos,
        mean_blur_time,
        mean_transfer_time
    );
    // (mean_blur_time, mean_transfer_time)
}

// take queue, get the image out, get the regions, blur the regions, put the image back
#[derive(Debug, Clone, Copy)]
pub(crate) enum BlurCondition {
    BlurKnown,
    BlurUnknown,
}

pub fn get_bbox(tuple: &Tuple, field: &HabString) -> [i64; 4] {
    match tuple.get(field) {
        Some(HabValue::ShapeBuffer(v)) => {
            let [x1, y1, x2, y2] = v.as_slice() else {
                error!("individual box bound field not a shape buffer of length 4 in tuple {:?} with img_id={:?}", tuple.id(), tuple.get(ORIGINAL_IMAGE_ID_FIELD));
                return [i64::MIN; 4];
            };
            [*x1 as i64, *y1 as i64, *x2 as i64, *y2 as i64]
        }
        Some(HabValue::IntBuffer(v)) => {
            let [x1, y1, x2, y2] = v.as_slice() else {
                error!("individual box bound field not an integer buffer of length 4 in tuple {:?} with img_id={:?}", tuple.id(), tuple.get(ORIGINAL_IMAGE_ID_FIELD));
                return [i64::MIN; 4];
            };
            [*x1 as i64, *y1 as i64, *x2 as i64, *y2 as i64]
        }
        Some(v) => {
            error!("individual box bound field not a shape buffer or an integer buffer (it had type {:?}) in tuple {:?} with img_id={:?}", v.get_type(), tuple.id(), tuple.get(ORIGINAL_IMAGE_ID_FIELD));
            [i64::MIN; 4]
        }
        None => {
            error!(
                "individual box bound field not found in tuple {:?} with img_id={:?}",
                tuple.id(),
                tuple.get(ORIGINAL_IMAGE_ID_FIELD)
            );
            [i64::MIN; 4]
        }
    }
}

pub(crate) fn aggregate_blur_regions(
    windowed_data: &mut Queue<Tuple>,
    blur_condition: BlurCondition,
) -> AggregationResult {
    let face_aggregate_start = std::time::Instant::now();
    if let Some(latest_tuple) = windowed_data.last() {
        let window_len = windowed_data.len();
        let tuple_id = latest_tuple.id();
        let match_ids_null = latest_tuple.get("match_ids").map_or(true, |v| v.is_null());
        let image_id = latest_tuple
            .get(ORIGINAL_IMAGE_ID_FIELD)
            .and_then(|v| v.as_string())
            .map(|s| s.as_str())
            .unwrap_or("<unknown image id>");
        let match_ids_len = if let Some(v) = latest_tuple.get("match_ids") {
            if let Some(l) = v.as_list() {
                l.len() as i64
            } else {
                -1
            }
        } else {
            -2
        };
        let box_id = latest_tuple
            .get(INDIVIDUAL_BOX_ID_FIELD)
            .and_then(|v| v.as_integer())
            .unwrap_or(-1);
        let log_location = "enter_face_aggregate".to_raw_key();
        let [x1, y1, x2, y2] = get_bbox(&latest_tuple, &INDIVIDUAL_BOX_BOUND_FIELD.into());
        let aux_data = Some(HashMap::from([
            (
                "latest_tuple_id".to_raw_key(),
                LimitedHabValue::Integer(tuple_id as _),
            ),
            (
                "image_id".to_raw_key(),
                LimitedHabValue::String(image_id.to_key()),
            ),
            (
                "match_ids_null".to_raw_key(),
                LimitedHabValue::Integer(match_ids_null as _),
            ),
            (
                "match_ids_len".to_raw_key(),
                LimitedHabValue::Integer(match_ids_len),
            ),
            ("box_id".to_raw_key(), LimitedHabValue::Integer(box_id as _)),
            (
                "num_tuples_in_window".to_raw_key(),
                LimitedHabValue::Integer(window_len as _),
            ),
            ("bbox_x1".to_raw_key(), LimitedHabValue::Integer(x1)),
            ("bbox_y1".to_raw_key(), LimitedHabValue::Integer(y1)),
            ("bbox_x2".to_raw_key(), LimitedHabValue::Integer(x2)),
            ("bbox_y2".to_raw_key(), LimitedHabValue::Integer(y2)),
        ]));
        debug!(
            "face aggregate window has {window_len} tuples, latest data has tuple_id={tuple_id}, match_ids_is_null={match_ids_null}, match_ids_len={match_ids_len}, bbox=[{x1}, {y1}, {x2}, {y2}], box_id={box_id}, image_id={image_id:?}"
        );
        if let Err(e) =
            watershed_shared::global_logger::log_data(tuple_id as _, log_location, aux_data)
        {
            for err in e {
                error!("failed to log face aggregate: {err}");
            }
        }
    } else {
        error!("windowed data should have at least one tuple");
    }
    let max_face_count = windowed_data
        .iter()
        .map(|tuple| {
            let Some(face_count) = tuple.get(FACE_COUNT_FIELD) else {
                warn!(
                    "face count field not found in tuple {:?}. fields that we did find: {:?}",
                    tuple.id(),
                    tuple.keys().collect::<Vec<_>>()
                );
                return 0;
            };
            face_count.as_integer().unwrap_or_default()
        })
        .max()
        .unwrap_or(0);
    let image_id_debug = windowed_data
        .get(0)
        .and_then(|t| t.get(ORIGINAL_IMAGE_ID_FIELD));
    if max_face_count == 0 {}
    if windowed_data.len() < max_face_count as usize {
        debug!(
            "not all faces detected yet. we expect {} faces, but image id {:?} only has {} tuples so far",
            max_face_count,
            image_id_debug,
            windowed_data.len()
        );
        return AggregationResult {
            emit: None,
            is_finished: false,
        };
    }

    let mut matches_detected = 0;
    for tuple in windowed_data.iter() {
        let Some(matches) = tuple.get("match_ids") else {
            continue;
        };
        let matches = match matches {
            HabValue::List(l) => Some(l.len()),
            HabValue::String(_) | HabValue::Integer(_) => Some(1),
            HabValue::Null => None,
            _ => {
                error!(
                    "match_ids field not a list in tuple {:?} with img_id={:?}",
                    tuple.id(),
                    tuple.get(ORIGINAL_IMAGE_ID_FIELD)
                );
                None
            }
        };
        if let Some(num_matches) = matches {
            if num_matches > 0 {
                matches_detected += 1;
            }
        }
    }

    // we know we're done if we can use the first image that we find
    // if we never find one, then we have encoutnered an error because we got all the tuples but lost the image along the way
    let (extracted_tuple, idx) = 'extract_image: {
        for (idx, tuple) in windowed_data.iter_mut().enumerate() {
            let Some(image) = tuple.get(ORIGINAL_IMAGE_FIELD) else {
                error!("original image field not found for tuple {:?}", tuple.id());
                continue;
            };
            let len = match image {
                HabValue::ByteBuffer(image) => image.len(),
                HabValue::SharedArrayU8(image) => image.0.len(),
                HabValue::SharedArrayF32(image) => image.0.len(),
                _ => {
                    error!(
                        "original image field not a byte buffer or an array for tuple {:?}",
                        tuple.id()
                    );
                    continue;
                }
            };

            if len == 0 {
                // error!("original image field is empty");
                continue;
            }
            break 'extract_image (tuple, idx);
        }

        error!("we received the expected number of tuples, but we didn't find a tuple that had the original image");
        return AggregationResult {
            emit: None,
            is_finished: true,
        };
    };

    let expected_matches = match extracted_tuple.get(EXPECTED_MATCHES_FIELD) {
        Some(HabValue::Integer(i)) if *i >= 0 => *i as i32,
        Some(HabValue::Integer(i)) => {
            warn!(
                "expected matches field is negative in tuple {:?} with img_id={:?}",
                extracted_tuple.id(),
                extracted_tuple.get(ORIGINAL_IMAGE_ID_FIELD)
            );
            *i
        }
        _ => {
            error!(
                "expected matches field not found in tuple {:?} with img_id={:?}",
                extracted_tuple.id(),
                extracted_tuple.get(ORIGINAL_IMAGE_ID_FIELD)
            );
            i32::MIN
        }
    };

    debug!(
        "image id {:?} has received all {} expected tuples.expected {} matches, actually found {} matches",
        // image_id_debug, max_face_count, expected_matches, matches_detected
        extracted_tuple.get(ORIGINAL_IMAGE_ID_FIELD), max_face_count, expected_matches, matches_detected
    );
    'record_matches_found: {
        let log_location = "all_faces_received".to_raw_key();
        // let aux_data = watershed_shared::ArrayMap::from([
        let aux_data = Some([
            (
                FACE_COUNT_FIELD.to_raw_key(),
                LimitedHabValue::Integer(max_face_count as _),
            ),
            (
                ORIGINAL_IMAGE_ID_FIELD.to_raw_key(),
                LimitedHabValue::String(
                    extracted_tuple
                        .get(ORIGINAL_IMAGE_ID_FIELD)
                        .and_then(|v| v.as_string().map(|s| s.to_key()))
                        .unwrap_or_else(|| "unknown".to_key().into()),
                ),
            ),
            (
                "matches_detected".to_raw_key(),
                LimitedHabValue::Integer(matches_detected as _),
            ),
            (
                "expected_matches".to_raw_key(),
                LimitedHabValue::Integer(expected_matches as _),
            ),
        ]);
        let tuple_id = extracted_tuple.id();
        if let Err(e) =
            watershed_shared::global_logger::log_data(tuple_id as _, log_location, aux_data)
        {
            for err in e {
                error!("failed to log all faces received: {err}");
            }
            break 'record_matches_found;
        }
    }
    let mut extracted_image = extract_image_from_tuple(
        extracted_tuple,
        ORIGINAL_IMAGE_SHAPE_FIELD.into(),
        ORIGINAL_IMAGE_FIELD.into(),
    );

    let blur_predicate = match blur_condition {
        BlurCondition::BlurKnown => blur_known_face_pred,
        BlurCondition::BlurUnknown => blur_unknown_face_pred,
    };
    let num_regions = windowed_data.len();
    let regions = get_blur_regions(
        windowed_data.as_slice(),
        INDIVIDUAL_BOX_BOUND_FIELD.into(),
        blur_predicate,
    );

    // blur_regions(&mut extracted_image, regions);
    const BLUR_KERNEL_SIZE: usize = 16;
    const BLUR_STRIDE_SIZE: usize = 16;
    const BLUR_INTERNAL_STRIDE_SIZE: usize = 4;
    box_blur_regions::<BLUR_KERNEL_SIZE, BLUR_STRIDE_SIZE, BLUR_INTERNAL_STRIDE_SIZE>(
        &mut extracted_image,
        regions,
    );

    let mut output_tuple = get_tuple();
    // this original image doesn't make sense to do because it has already been taken out
    // this either needs to be removed/commented out or it needs to be changed to copy the image and have both the original and blurred versions in the final
    let Some(tuple_for_id) = windowed_data.get(idx) else {
        error!("failed to get tuple for id");
        panic!("failed to get tuple for id");
    };
    let Some(original_image_id_value) = tuple_for_id.get(ORIGINAL_IMAGE_ID_FIELD) else {
        error!("original image id field not found");
        panic!("original image id field not found");
    };
    output_tuple.insert(
        ORIGINAL_IMAGE_ID_FIELD.into(),
        original_image_id_value.clone(),
    );
    let Some(original_image_id_int_value) = tuple_for_id.get(ORIGINAL_IMAGE_ID_INT_FIELD) else {
        error!("original image id int field not found");
        panic!("original image id int field not found");
    };
    output_tuple.insert(
        ORIGINAL_IMAGE_ID_INT_FIELD.into(),
        original_image_id_int_value.clone(),
    );
    insert_image_into_tuple(
        &mut output_tuple,
        ORIGINAL_IMAGE_SHAPE_FIELD.into(),
        BLURRED_IMAGE_FIELD.into(),
        extracted_image,
    );
    let mut tuple_vec = get_tuple_vec();
    tuple_vec.push(output_tuple);
    let face_aggregate_elapsed_ms = face_aggregate_start.elapsed().as_nanos() as f64 / 1_000_000.0;
    info!(
        "aggregated {} tuples for blurring in {:.4} ms",
        num_regions, face_aggregate_elapsed_ms,
    );
    AggregationResult {
        emit: Some(tuple_vec),
        is_finished: true,
    }
}

fn blur_known_face_pred(tuple: &Tuple) -> bool {
    let tuple_id = tuple.id();
    let Some(matches) = tuple.get("match_ids") else {
        return false;
    };
    let Some(box_num) = tuple.get(INDIVIDUAL_BOX_ID_FIELD) else {
        error!(
            "my_box_number field not found for tuple {:?} (img {:?})",
            tuple.id(),
            tuple.get(ORIGINAL_IMAGE_ID_FIELD)
        );
        return true;
    };
    match matches {
        HabValue::List(matches) => {
            if matches.len() == 0 {
                trace!("match_ids field is a list but has no elements. we take this to mean it has no valid chromadb matches. skipping");
                debug!(
                    "no matches for box {:?} in img {:?}",
                    box_num,
                    tuple.get(ORIGINAL_IMAGE_ID_FIELD)
                );
                return false;
            }
            debug!(
                "blurring known face for box {:?} in img {:?}",
                box_num,
                tuple.get(ORIGINAL_IMAGE_ID_FIELD)
            );
            trace!(
                "there is at least one confident match for box {:?} in tuple {} (img {:?}). blurring",
                box_num, tuple.id(), tuple.get(ORIGINAL_IMAGE_ID_FIELD)
            );

            'log_match: {
                // use global logger
                let log_location = "one_match".to_raw_key();
                let aux_data = NO_AUX_DATA;
                if let Err(e) =
                    watershed_shared::global_logger::log_data(tuple_id, log_location, aux_data)
                {
                    let img_field = tuple.get(ORIGINAL_IMAGE_ID_FIELD);
                    // import trait for write macro
                    use std::fmt::Write;
                    let mut log_str = format!(
                        "{} failures when logging no matches for box {:?} in img {:?}",
                        e.len(),
                        box_num,
                        img_field
                    );
                    for err in e {
                        if let Err(e) = write!(log_str, "\n{err}") {
                            error!("failed to write error to log string: {e}");
                            break 'log_match;
                        }
                    }
                    error!("{log_str}");
                }
            }
            true
        }
        HabValue::String(_) | HabValue::Integer(_) => {
            let Some(box_num) = tuple.get(INDIVIDUAL_BOX_ID_FIELD) else {
                error!("my_box_number field not found, but we did have a match");
                debug!(
                    "bad String match for box {:?} in img {:?}",
                    box_num,
                    tuple.get(ORIGINAL_IMAGE_ID_FIELD)
                );
                return true;
            };
            debug!(
                "blurring known face for box {:?} in img {:?}",
                box_num,
                tuple.get(ORIGINAL_IMAGE_ID_FIELD)
            );
            true
        }
        HabValue::Null => {
            trace!("match_ids field is null. we take this to mean it has no valid chromadb matches. skipping");

            'log_no_match: {
                // use global logger
                let log_location = "no_match".to_raw_key();
                let aux_data = NO_AUX_DATA;
                let img_field = tuple.get(ORIGINAL_IMAGE_ID_FIELD);
                if let Err(e) =
                    watershed_shared::global_logger::log_data(tuple_id, log_location, aux_data)
                {
                    // import trait for write macro
                    use std::fmt::Write;
                    let mut log_str = format!(
                        "{} failures when logging no matches for box {:?} in img {:?}",
                        e.len(),
                        box_num,
                        img_field
                    );
                    for err in e {
                        if let Err(e) = write!(log_str, "\n{err}") {
                            error!("failed to write error to log string : {e}");
                            break 'log_no_match;
                        }
                    }
                    error!("{log_str}");
                }
            }
            false
        }
        _ => {
            error!(
                "match_ids is type {:?}, which is not a list, string, or null. skipping",
                matches
            );
            false
        }
    }
}

fn blur_unknown_face_pred(tuple: &Tuple) -> bool {
    // negative case
    let Some(matches) = tuple.get("match_ids") else {
        warn!("match_ids field not found, blurring");
        return true;
    };
    match matches {
        HabValue::List(matches) => {
            if matches.len() == 0 {
                debug!("match_ids field is a list but has no elements. we take this to mean it has no valid chromadb matches. blurring");
                return true;
            }
            debug!("not blurring known face with multiple matches");
            false
        }
        HabValue::String(_match_id) => {
            let Some(_box_number) = tuple.get(INDIVIDUAL_BOX_ID_FIELD) else {
                error!("my_box_number field not found, but we did have a String match, so we are not blurring");
                return false;
            };
            debug!("not blurring known face");
            false
        }
        // we blur when we don't find anything to preserve privacy of uninvolved persons
        HabValue::Null => true,
        _ => {
            error!(
                "match_ids is type {:?}, which is not a list, string, or null. skipping blurring",
                matches
            );
            true
        }
    }
}

pub(crate) fn decode_embeddings_inline(
    val: &Py<PyAny>,
    mut original: Tuple,
    fields: &[HabString],
    out: &mut Vec<Tuple>,
) -> Option<usize> {
    if fields.len() != 1 {
        warn!(
            "embedding decoding expected 1 field, got {}. it will only be using the first",
            fields.len()
        );
    }
    let Ok(v): pyo3::PyResult<Vec<Vec<f32>>> = pyo3::Python::with_gil(|py| val.extract(py)) else {
        error!("failed to extract embeddings as list of list of floats");
        return None;
    };
    let expected_outputs = v.len();
    if expected_outputs == 0 {
        warn!("no embeddings found");
        return Some(0);
    }
    let copies_to_make = v.len().saturating_sub(1);
    out.push(get_tuple());
    for i in 1..(copies_to_make + 1) {
        let mut new_tuple = get_tuple();
        for (original_field, original_value) in original.iter() {
            if original_field == &fields[0] {
                continue;
            }
            // we need to check if it's a buffer so we don't copy an expensive image or ndarray
            let new_value = match original_value {
                HabValue::ByteBuffer(_)
                // shape buffers are expected to be small
                // | HabValue::ShapeBuffer(_)
                | HabValue::IntBuffer(_)
                =>{
                    HabValue::Null
                }
                _ => { original_value.clone() }
            };
            new_tuple.insert(original_field.clone(), new_value);
        }
        new_tuple.insert(
            fields[0].clone(),
            HabValue::List(v[i].iter().copied().map(|x| (x as f64).into()).collect()),
        );
        out.push(new_tuple);
    }

    original.insert(
        fields[0].clone(),
        HabValue::List(v[0].iter().copied().map(|x| (x as f64).into()).collect()),
    );
    std::mem::swap(&mut original, &mut out[0]);
    // original now holds our temporary from earlier
    return_tuple(original);
    Some(expected_outputs)
}

pub fn split_bbs(
    mut input: Tuple,
    resize_width: u32,
    resize_height: u32,
    preprocessing: impl Fn(image::DynamicImage, u32, u32) -> ndarray::ArrayD<f32>,
) -> Vec<Tuple> {
    let mut outputs = get_tuple_vec();
    let Some(bb_dims) = input.get(omz_utils::BOXES_SHAPE_FIELD) else {
        error!("bounding boxes field not found");
        return vec![];
    };
    let Some(bb_dims) = bb_dims.as_shape_buffer() else {
        error!("bounding boxes field not a shape buffer");
        return vec![];
    };
    if bb_dims.len() < 1 {
        error!("bounding boxes shape buffer has no dimensions");
        return vec![];
    }

    let mut faces_detected = bb_dims[0] as usize;
    let Some(bbs_buffer) = input.get(omz_utils::BOXES_BUFFER_FIELD) else {
        error!("bounding boxes field not found");
        return vec![];
    };
    let Some(bbs_buffer) = bbs_buffer.as_int_buffer() else {
        error!("bounding boxes field not a byte buffer");
        return vec![];
    };
    // let bb_buffer = bytemuck::cast_slice::<_, f32>(bbs_buffer);
    // omz postprocessing already makes i32s
    let mut bb_buffer = bbs_buffer;
    let Some(original_image_buffer) = input.get(ORIGINAL_IMAGE_FIELD) else {
        error!("original image field not found");
        return vec![];
    };
    let Some(original_image_buffer) = original_image_buffer.as_byte_buffer() else {
        error!("original image field not a byte buffer");
        return vec![];
    };
    let Some(original_image_shape) = input.get(ORIGINAL_IMAGE_SHAPE_FIELD) else {
        error!("original image shape field not found");
        return vec![];
    };
    let Some(original_image_shape) = original_image_shape.as_shape_buffer() else {
        error!("original image shape field not a shape buffer");
        return vec![];
    };
    let Some(image_id) = input.get(ORIGINAL_IMAGE_ID_FIELD) else {
        error!("original image id field not found");
        return vec![];
    };
    let Some(image_id) = image_id.as_string() else {
        error!("original image id field not a string");
        return vec![];
    };

    // reform image
    let Some(img) = image::RgbImage::from_raw(
        original_image_shape[0] as u32,
        original_image_shape[1] as u32,
        original_image_buffer.as_ref().to_vec(),
    ) else {
        // dimensions are wrong
        error!("dimensions are wrong for tuple with id {:?}. Expected shape {}x{}={}, but actual size was", input.id(), original_image_shape[0], original_image_shape[1], original_image_buffer.as_ref().len());
        return vec![];
    };

    let max_x = original_image_shape[0] as u32;
    let max_y = original_image_shape[1] as u32;

    let do_not_crop = std::env::var("NO_CROP").is_ok();
    let save_image_steps = std::env::var("LOG_IMAGE_STEPS").is_ok();
    let no_crop_buf;
    if do_not_crop {
        faces_detected = 1;
        no_crop_buf = [0, 0, max_x as i32, max_y as i32];
        bb_buffer = &no_crop_buf;
    }

    // dummy tuple
    outputs.push(get_tuple());
    let original_tuple_id = input.id();
    let original_img_id = image_id;
    let mut output_tuple_ids = smallvec::SmallVec::<[usize; 16]>::new();

    for box_index in 1..faces_detected {
        let mut new_tuple = get_tuple();
        let new_tuple_id = new_tuple.id();
        new_tuple.insert(
            INDIVIDUAL_BOX_ID_FIELD.into(),
            HabValue::Integer(box_index as i32),
        );
        new_tuple.insert(
            "bounding_boxes".into(),
            // bytemuck::cast_slice::<_, i32>(bs_buffer).to_vec(),
            HabValue::Null,
        );
        // image id
        new_tuple.insert(
            ORIGINAL_IMAGE_ID_FIELD.into(),
            HabValue::String(image_id.clone()),
        );

        let x1 = bb_buffer[box_index * 4];
        let y1 = bb_buffer[box_index * 4 + 1];
        let x2 = bb_buffer[box_index * 4 + 2];
        let y2 = bb_buffer[box_index * 4 + 3];
        let unclamped_box = [[x1, y1], [x2, y2]];

        debug!("resizing box {box_index}/{faces_detected} from image with id {image_id:?}. original dimensions=({max_x}, {max_y}), unclamped bounding box={unclamped_box:?}");
        if x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 {
            warn!("negative pixel found in unclamped box {unclamped_box:?} , which is box {box_index}/{faces_detected} from image with id {image_id:?}");
        }
        if x2 - x1 <= 0 || y2 - y1 <= 0 {
            error!("zero or negative area found in unclamped box {unclamped_box:?} , which is box {box_index}/{faces_detected} from image with id {image_id:?}");
        }

        let x1 = (bb_buffer[box_index * 4] as i32).clamp(0, max_x as i32);
        let y1 = (bb_buffer[box_index * 4 + 1] as i32).clamp(0, max_y as i32);
        let x2 = (bb_buffer[box_index * 4 + 2] as i32).clamp(0, max_x as i32);
        let y2 = (bb_buffer[box_index * 4 + 3] as i32).clamp(0, max_y as i32);
        new_tuple.insert(
            INDIVIDUAL_BOX_BOUND_FIELD.into(),
            HabValue::IntBuffer(vec![x1, y1, x2, y2]),
        );
        new_tuple.insert(
            INDIVIDUAL_BOX_ID_FIELD.into(),
            HabValue::Integer(box_index as i32),
        );
        new_tuple.insert(
            FACES_DETECTED_FIELD.into(),
            HabValue::Integer(faces_detected as _),
        );

        let cropped_img = image::DynamicImage::ImageRgb8(
            img.view(
                x1 as u32,
                y1 as u32,
                x2 as u32 - x1 as u32,
                y2 as u32 - y1 as u32,
            )
            .to_image(),
        );
        if save_image_steps {
            if let Err(e) = cropped_img.save(format!("cropped_box_{box_index}__{image_id}")) {
                error!(
                    "could not save cropped image for box {box_index} of image {image_id}: {e:?}"
                );
            }
        }

        let processed_input = preprocessing(cropped_img, resize_width, resize_height);
        let processed_input_shape = processed_input.shape().to_owned();
        debug!(
            "original image id {} created cropped image with shape {:?}",
            image_id, processed_input_shape
        );
        let (processed_input, Some(0) | None) = processed_input.into_raw_vec_and_offset() else {
            error!("failed to get raw vec and offset");
            return vec![];
        };
        new_tuple.insert(
            "cropped_image".into(),
            HabValue::IntBuffer(bytemuck::cast_vec(processed_input)),
        );
        new_tuple.insert(
            "cropped_image_shape".into(),
            HabValue::ShapeBuffer(processed_input_shape.clone()),
        );
        new_tuple.insert(
            "resized_shape".into(),
            HabValue::ShapeBuffer(processed_input_shape),
        );

        new_tuple.insert(
            ORIGINAL_IMAGE_FIELD.into(),
            // original_image_buffer.clone(),
            HabValue::Null,
        );
        new_tuple.insert(ORIGINAL_IMAGE_SHAPE_FIELD.into(), HabValue::Null);
        debug!(
            "original image id {image_id:?} created new tuple with id {:?} to house bounding box #{box_index} with bounds (({x1}, {y1}), ({x2}, {y2}))",
            new_tuple.id()
        );

        'log_create_box_tuple: {
            use watershed_shared::global_logger;
            let log_location = "split_box_tuple".to_raw_key();
            let aux_data = Some(HashMap::from([
                (
                    "image_id".to_raw_key(),
                    LimitedHabValue::String(image_id.to_key()),
                ),
                (
                    "box_index".to_raw_key(),
                    LimitedHabValue::Integer(box_index as _),
                ),
                ("x1".to_raw_key(), LimitedHabValue::Integer(x1 as _)),
                ("y1".to_raw_key(), LimitedHabValue::Integer(y1 as _)),
                ("x2".to_raw_key(), LimitedHabValue::Integer(x2 as _)),
                ("y2".to_raw_key(), LimitedHabValue::Integer(y2 as _)),
            ]));
            if let Err(e) = global_logger::log_data(new_tuple_id, log_location, aux_data) {
                for err in e {
                    error!("failed to log box tuple creation: {err}");
                }
                break 'log_create_box_tuple;
            }
        }

        outputs.push(new_tuple);
    }

    let (my_box_number, my_box_bounds, cropped_image, cropped_image_shape, resized_shape) =
        if faces_detected != 0 {
            let box_index = 0;
            let x1 = bb_buffer[box_index * 4];
            let y1 = bb_buffer[box_index * 4 + 1];
            let x2 = bb_buffer[box_index * 4 + 2];
            let y2 = bb_buffer[box_index * 4 + 3];
            let unclamped_box = [[x1, y1], [x2, y2]];

            let new_tuple_id = input.id();
            'log_create_box_tuple: {
                use watershed_shared::global_logger;
                let log_location = "split_box_tuple".to_raw_key();
                let aux_data = Some(HashMap::from([
                    (
                        "image_id".to_raw_key(),
                        LimitedHabValue::String(image_id.to_key()),
                    ),
                    (
                        "box_index".to_raw_key(),
                        LimitedHabValue::Integer(box_index as _),
                    ),
                    ("x1".to_raw_key(), LimitedHabValue::Integer(x1 as _)),
                    ("y1".to_raw_key(), LimitedHabValue::Integer(y1 as _)),
                    ("x2".to_raw_key(), LimitedHabValue::Integer(x2 as _)),
                    ("y2".to_raw_key(), LimitedHabValue::Integer(y2 as _)),
                ]));
                if let Err(e) = global_logger::log_data(new_tuple_id, log_location, aux_data) {
                    for err in e {
                        error!("failed to log box tuple creation: {err}");
                    }
                    break 'log_create_box_tuple;
                }
            }

            debug!("resizing box {box_index}/{faces_detected} from image with id {image_id:?}. original dimensions=({max_x}, {max_y}), unclamped bounding box={unclamped_box:?}");
            if x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 {
                warn!("negative pixel found in unclamped box {unclamped_box:?} , which is box {box_index}/{faces_detected} from image with id {image_id:?}");
            }
            if x2 - x1 <= 0 || y2 - y1 <= 0 {
                error!("zero or negative area found in unclamped box {unclamped_box:?} , which is box {box_index}/{faces_detected} from image with id {image_id:?}");
            }

            // get the first bounding box and use it to replace the dummy tuple
            // modify the original tuple
            let x1 = (bb_buffer[0] as i32).clamp(0, max_x as i32);
            let y1 = (bb_buffer[1] as i32).clamp(0, max_y as i32);
            let x2 = (bb_buffer[2] as i32).clamp(0, max_x as i32);
            let y2 = (bb_buffer[3] as i32).clamp(0, max_y as i32);
            let cropped_img = image::DynamicImage::ImageRgb8(
                img.view(
                    x1 as u32,
                    y1 as u32,
                    x2 as u32 - x1 as u32,
                    y2 as u32 - y1 as u32,
                )
                .to_image(),
            );
            if save_image_steps {
                if let Err(e) = cropped_img.save(format!("cropped_box_{box_index}__{image_id}")) {
                    error!(
                        "could not save cropped image for box {box_index} of image {image_id}: {e:?}"
                    );
                }
            }

            let processed_input = preprocessing(cropped_img, resize_width, resize_height);
            let processed_input_shape = processed_input.shape().to_owned();
            debug!(
                "original image id {} created cropped image with shape {:?}",
                image_id, processed_input_shape
            );

            let (processed_input, Some(0) | None) = processed_input.into_raw_vec_and_offset()
            else {
                error!("failed to get raw vec and offset");
                return vec![];
            };
            let bb_bounds = vec![x1, y1, x2, y2];
            let my_box_bounds = HabValue::IntBuffer(bb_bounds);
            let my_box_number = HabValue::Integer(0);
            let cropped_image = HabValue::IntBuffer(bytemuck::cast_vec(processed_input));
            let cropped_image_shape = HabValue::ShapeBuffer(processed_input_shape.clone());
            let resized_shape = HabValue::ShapeBuffer(processed_input_shape);
            debug!(
                "original image id {:?} re-used tuple with id {:?} to house bounding box #0 with bounds (({x1}, {y1}), ({x2}, {y2}))",
                image_id,
                input.id()
            );
            (
                my_box_number,
                my_box_bounds,
                cropped_image,
                cropped_image_shape,
                resized_shape,
            )
        } else {
            // no faces detected, so we just use the original image
            let my_box_number = HabValue::Null;
            let my_box_bounds = HabValue::Null;
            let cropped_image = HabValue::Null;
            let cropped_image_shape = HabValue::Null;
            let resized_shape = HabValue::Null;
            debug!(
                "no faces found for original image id {:?} ; setting fields to null",
                image_id
            );
            (
                my_box_number,
                my_box_bounds,
                cropped_image,
                cropped_image_shape,
                resized_shape,
            )
        };
    drop(original_image_buffer);
    input.insert(INDIVIDUAL_BOX_BOUND_FIELD.into(), my_box_bounds);
    input.insert(INDIVIDUAL_BOX_ID_FIELD.into(), my_box_number);
    input.insert(
        FACES_DETECTED_FIELD.into(),
        HabValue::Integer(if do_not_crop { 1 } else { faces_detected as _ }),
    );
    input.insert("cropped_image".into(), cropped_image);
    input.insert("resized_shape".into(), resized_shape);
    input.insert("cropped_image_shape".into(), cropped_image_shape);

    std::mem::swap(&mut input, &mut outputs[0]);

    outputs
}

pub fn split_bbs_facenet(input: Tuple) -> Vec<Tuple> {
    split_bbs(input, 160, 160, preprocess_facenet)
}

pub fn split_bbs_insightface(input: Tuple) -> Vec<Tuple> {
    split_bbs(input, 112, 112, preprocess_insightface)
}

pub fn split_bbs_before_scheduling(mut input: Tuple) -> Vec<Tuple> {
    let mut outputs = get_tuple_vec();
    let Some(bb_dims) = input.get(omz_utils::BOXES_SHAPE_FIELD) else {
        error!("bounding boxes field not found");
        return vec![];
    };
    let Some(bb_dims) = bb_dims.as_shape_buffer() else {
        error!("bounding boxes field not a shape buffer");
        return vec![];
    };
    if bb_dims.len() < 1 {
        error!("bounding boxes shape buffer has no dimensions");
        return vec![];
    }

    let mut faces_detected = bb_dims[0] as usize;
    let Some(bbs_buffer) = input.get(omz_utils::BOXES_BUFFER_FIELD) else {
        error!("bounding boxes field not found");
        return vec![];
    };
    let Some(bbs_buffer) = bbs_buffer.as_int_buffer() else {
        error!("bounding boxes field not a byte buffer");
        return vec![];
    };
    // let bb_buffer = bytemuck::cast_slice::<_, f32>(bbs_buffer);
    // omz postprocessing already makes i32s
    let mut bb_buffer = bbs_buffer;
    let Some(original_image_buffer) = input.get(ORIGINAL_IMAGE_FIELD) else {
        error!("original image field not found");
        return vec![];
    };
    enum CopyMaker<'a> {
        ByteBuffer(&'a [u8]),
        SharedArrayU8(watershed_shared::ws_types::ArcArrayD<u8>),
        // SharedArrayF32(watershed_shared::SharedArrayF32),
    }
    let original_image_buffer = match original_image_buffer {
        HabValue::ByteBuffer(buf) => CopyMaker::ByteBuffer(buf.as_ref()),
        HabValue::SharedArrayU8(buf) => CopyMaker::SharedArrayU8(buf.0.clone()),
        // HabValue::SharedArrayF32(buf) => CopyMaker::SharedArrayF32(buf.clone()),
        _ => {
            error!("original image field not a byte buffer or shared array");
            return vec![];
        }
    };
    impl CopyMaker<'_> {
        fn make_habvalue(&self) -> HabValue {
            match self {
                CopyMaker::ByteBuffer(buf) => HabValue::ByteBuffer(buf.to_vec()),
                CopyMaker::SharedArrayU8(buf) => {
                    HabValue::SharedArrayU8(watershed_shared::ws_types::SharedU8Array(buf.clone()))
                } // CopyMaker::SharedArrayF32(buf) => HabValue::SharedArrayF32(buf.clone()),
            }
        }
    }

    let Some(original_image_shape) = input.get(ORIGINAL_IMAGE_SHAPE_FIELD) else {
        error!("original image shape field not found");
        return vec![];
    };
    let Some(original_image_shape) = original_image_shape.as_shape_buffer() else {
        error!("original image shape field not a shape buffer");
        return vec![];
    };
    let img_buffer_shape_copy = original_image_shape.as_ref().to_vec();

    let Some(image_id) = input.get(ORIGINAL_IMAGE_ID_FIELD) else {
        error!("original image id field not found");
        return vec![];
    };
    let Some(image_id) = image_id.as_string() else {
        error!("original image id field not a string");
        return vec![];
    };
    let Some(image_id_int) = input.get(ORIGINAL_IMAGE_ID_INT_FIELD) else {
        error!("original image id field not found");
        return vec![];
    };
    let Some(image_id_int) = image_id_int.as_integer() else {
        error!("original image id int field not an integer");
        return vec![];
    };
    let Some(expected_matches) = input.get(EXPECTED_MATCHES_FIELD) else {
        error!("expected matches field not found");
        return vec![];
    };
    let Some(mut expected_matches) = expected_matches.as_integer() else {
        error!("expected matches field not an integer");
        return vec![];
    };

    let max_x = original_image_shape[0] as u32;
    let max_y = original_image_shape[1] as u32;

    static DO_NOT_CROP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let do_not_crop = *DO_NOT_CROP.get_or_init(|| std::env::var("NO_CROP").is_ok());
    let no_crop_buf;
    if do_not_crop {
        faces_detected = 1;
        no_crop_buf = [0, 0, max_x as i32, max_y as i32];
        bb_buffer = &no_crop_buf;
    }
    static NUM_DUPLICATE_BOXES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let num_duplicate_boxes = *NUM_DUPLICATE_BOXES.get_or_init(|| {
        let Ok(count) = std::env::var("DUPLICATE_BOXES") else {
            debug!("DUPLICATE_BOXES not set, defaulting to 1 (no repeats)");
            return 1;
        };
        let Ok(count) = count.parse::<usize>() else {
            error!("DUPLICATE_BOXES is set to {count:?}, not a valid usize, defaulting to 1 (no repeats)");
            return 1;
        };
        if count < 1 {
            error!("DUPLICATE_BOXES is set to {count}, which is less than 1, defaulting to 1 (no repeats)");
            return 1;
        }
        debug!("DUPLICATE_BOXES is set to {count}, duplicating each bounding box that many times");
        count
    });
    let mut duplicate_box_buffer: Vec<i32>;
    if num_duplicate_boxes > 1 {
        const BOUNDING_BOX_DIMS: usize = 4;
        duplicate_box_buffer =
            Vec::with_capacity(faces_detected * BOUNDING_BOX_DIMS * num_duplicate_boxes);
        for _ in 0..num_duplicate_boxes {
            duplicate_box_buffer.extend_from_slice(&bb_buffer[..]);
        }
        bb_buffer = &duplicate_box_buffer;
        faces_detected *= num_duplicate_boxes;
        expected_matches *= num_duplicate_boxes as i32;
        info!("duplicating each bounding box {num_duplicate_boxes} times, for a total of {faces_detected} boxes");
    } else {
        debug!("not duplicating bounding boxes");
        duplicate_box_buffer = bb_buffer.to_vec();
    }

    // dummy tuple
    outputs.push(get_tuple());
    let mut output_tuple_ids = smallvec::SmallVec::<[usize; 16]>::new();
    output_tuple_ids.push(input.id());

    for box_index in 1..faces_detected {
        let mut new_tuple = get_tuple();
        new_tuple.mirror_time_created(&input);
        let new_tuple_id = new_tuple.id();
        new_tuple.insert(
            INDIVIDUAL_BOX_ID_FIELD.into(),
            HabValue::Integer(box_index as i32),
        );
        new_tuple.insert(
            "bounding_boxes".into(),
            // bytemuck::cast_slice::<_, i32>(bs_buffer).to_vec(),
            HabValue::Null,
        );
        // image id
        new_tuple.insert(
            ORIGINAL_IMAGE_ID_FIELD.into(),
            HabValue::String(image_id.clone()),
        );
        new_tuple.insert(
            ORIGINAL_IMAGE_ID_INT_FIELD.into(),
            HabValue::Integer(image_id_int),
        );
        new_tuple.insert(
            EXPECTED_MATCHES_FIELD.into(),
            HabValue::Integer(expected_matches),
        );
        // set box buffer field
        new_tuple.insert(
            omz_utils::BOXES_BUFFER_FIELD.into(),
            HabValue::IntBuffer(duplicate_box_buffer.clone()),
        );

        let x1 = bb_buffer[box_index * 4];
        let y1 = bb_buffer[box_index * 4 + 1];
        let x2 = bb_buffer[box_index * 4 + 2];
        let y2 = bb_buffer[box_index * 4 + 3];
        let unclamped_box = [[x1, y1], [x2, y2]];

        debug!("resizing box {box_index}/{faces_detected} from image with id {image_id:?}. original dimensions=({max_x}, {max_y}), unclamped bounding box={unclamped_box:?}");
        if x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 {
            warn!("negative pixel location found in unclamped box {unclamped_box:?} , which is box {box_index}/{faces_detected} from image with id {image_id:?}");
        }
        if x2 - x1 <= 0 || y2 - y1 <= 0 {
            error!("zero or negative area found in unclamped box {unclamped_box:?} , which is box {box_index}/{faces_detected} from image with id {image_id:?}");
        }

        let x1 = (bb_buffer[box_index * 4] as i32).clamp(0, max_x as i32);
        let y1 = (bb_buffer[box_index * 4 + 1] as i32).clamp(0, max_y as i32);
        let x2 = (bb_buffer[box_index * 4 + 2] as i32).clamp(0, max_x as i32);
        let y2 = (bb_buffer[box_index * 4 + 3] as i32).clamp(0, max_y as i32);
        new_tuple.insert(
            INDIVIDUAL_BOX_BOUND_FIELD.into(),
            HabValue::IntBuffer(vec![x1, y1, x2, y2]),
        );
        new_tuple.insert(
            INDIVIDUAL_BOX_ID_FIELD.into(),
            HabValue::Integer(box_index as i32),
        );
        new_tuple.insert(
            FACES_DETECTED_FIELD.into(),
            HabValue::Integer(faces_detected as _),
        );

        let new_height_float = (y2 - y1) as f64;
        let new_width_float = (x2 - x1) as f64;
        let new_area_float = new_height_float * new_width_float;
        let new_hw_ratio_float = new_height_float / new_width_float;
        let new_height_float = HabValue::from(new_height_float);
        let new_width_float = HabValue::from(new_width_float);
        let new_area_float = HabValue::from(new_area_float);
        let new_hw_ratio_float = HabValue::from(new_hw_ratio_float);

        new_tuple.insert("new_height_float".into(), new_height_float);
        new_tuple.insert("new_width_float".into(), new_width_float);
        new_tuple.insert("new_area_float".into(), new_area_float);
        new_tuple.insert("new_hw_ratio_float".into(), new_hw_ratio_float);

        new_tuple.insert(
            ORIGINAL_IMAGE_FIELD.into(),
            original_image_buffer.make_habvalue(),
        );
        new_tuple.insert(
            ORIGINAL_IMAGE_SHAPE_FIELD.into(),
            HabValue::ShapeBuffer(img_buffer_shape_copy.clone()),
        );
        debug!(
            "original image id {image_id:?} created new tuple with id {:?} to house bounding box #{box_index} with bounds (({x1}, {y1}), ({x2}, {y2}))",
            new_tuple.id()
        );

        'log_create_box_tuple: {
            use watershed_shared::global_logger;
            let log_location = "split_box_tuple".to_raw_key();
            let aux_data = Some(HashMap::from([
                (
                    "image_id".to_raw_key(),
                    LimitedHabValue::String(image_id.to_key()),
                ),
                (
                    "box_index".to_raw_key(),
                    LimitedHabValue::Integer(box_index as _),
                ),
                ("x1".to_raw_key(), LimitedHabValue::Integer(x1 as _)),
                ("y1".to_raw_key(), LimitedHabValue::Integer(y1 as _)),
                ("x2".to_raw_key(), LimitedHabValue::Integer(x2 as _)),
                ("y2".to_raw_key(), LimitedHabValue::Integer(y2 as _)),
            ]));
            if let Err(e) = global_logger::log_data(new_tuple_id, log_location, aux_data) {
                for err in e {
                    error!("failed to log box tuple creation: {err}");
                }
                break 'log_create_box_tuple;
            }
        }

        output_tuple_ids.push(new_tuple_id);
        outputs.push(new_tuple);
    }

    let (
        my_box_number,
        my_box_bounds,
        new_height_float,
        new_width_float,
        new_area_float,
        new_hw_ratio_float,
        //   cropped_image,
        //    cropped_image_shape,
        //     resized_shape,
    ) = if faces_detected != 0 {
        let box_index = 0;
        let x1 = bb_buffer[box_index * 4];
        let y1 = bb_buffer[box_index * 4 + 1];
        let x2 = bb_buffer[box_index * 4 + 2];
        let y2 = bb_buffer[box_index * 4 + 3];
        let unclamped_box = [[x1, y1], [x2, y2]];

        let new_tuple_id = input.id();
        'log_create_box_tuple: {
            use watershed_shared::global_logger;
            let log_location = "split_box_tuple".to_raw_key();
            let aux_data = Some(HashMap::from([
                (
                    "image_id".to_raw_key(),
                    LimitedHabValue::String(image_id.to_key()),
                ),
                (
                    "box_index".to_raw_key(),
                    LimitedHabValue::Integer(box_index as _),
                ),
                ("x1".to_raw_key(), LimitedHabValue::Integer(x1 as _)),
                ("y1".to_raw_key(), LimitedHabValue::Integer(y1 as _)),
                ("x2".to_raw_key(), LimitedHabValue::Integer(x2 as _)),
                ("y2".to_raw_key(), LimitedHabValue::Integer(y2 as _)),
            ]));
            if let Err(e) = global_logger::log_data(new_tuple_id, log_location, aux_data) {
                for err in e {
                    error!("failed to log box tuple creation: {err}");
                }
                break 'log_create_box_tuple;
            }
        }

        debug!("resizing box {box_index}/{faces_detected} from image with id {image_id:?}. original dimensions=({max_x}, {max_y}), unclamped bounding box={unclamped_box:?}");
        if x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 {
            warn!("negative pixel location found in unclamped box {unclamped_box:?} , which is box {box_index}/{faces_detected} from image with id {image_id:?}");
        }
        if x2 - x1 <= 0 || y2 - y1 <= 0 {
            error!("zero or negative area found in unclamped box {unclamped_box:?} , which is box {box_index}/{faces_detected} from image with id {image_id:?}");
        }

        // get the first bounding box and use it to replace the dummy tuple
        // modify the original tuple
        let x1 = (bb_buffer[0] as i32).clamp(0, max_x as i32);
        let y1 = (bb_buffer[1] as i32).clamp(0, max_y as i32);
        let x2 = (bb_buffer[2] as i32).clamp(0, max_x as i32);
        let y2 = (bb_buffer[3] as i32).clamp(0, max_y as i32);

        let bb_bounds = vec![x1, y1, x2, y2];
        let new_height_float = (y2 - y1) as f64;
        let new_width_float = (x2 - x1) as f64;
        let new_area_float = new_height_float * new_width_float;
        let new_hw_ratio_float = new_height_float / new_width_float;
        let new_height_float = HabValue::from(new_height_float);
        let new_width_float = HabValue::from(new_width_float);
        let new_area_float = HabValue::from(new_area_float);
        let new_hw_ratio_float = HabValue::from(new_hw_ratio_float);

        let my_box_bounds = HabValue::IntBuffer(bb_bounds);
        let my_box_number = HabValue::Integer(0);

        debug!(
                "original image id {:?} re-used tuple with id {:?} to house bounding box #0 with bounds (({x1}, {y1}), ({x2}, {y2}))",
                image_id,
                input.id()
            );
        (
            my_box_number,
            my_box_bounds,
            new_height_float,
            new_width_float,
            new_area_float,
            new_hw_ratio_float,
        )
    } else {
        // no faces detected, so we just use the original image
        let my_box_number = HabValue::Null;
        let my_box_bounds = HabValue::Null;
        let new_height_float = HabValue::Null;
        let new_width_float = HabValue::Null;
        let new_area_float = HabValue::Null;
        let new_hw_ratio_float = HabValue::Null;

        debug!(
            "no faces found for original image id {:?} ; setting fields to null",
            image_id
        );
        (
            my_box_number,
            my_box_bounds,
            new_height_float,
            new_width_float,
            new_area_float,
            new_hw_ratio_float,
        )
    };
    drop(original_image_buffer);
    input.insert(INDIVIDUAL_BOX_BOUND_FIELD.into(), my_box_bounds);
    input.insert(INDIVIDUAL_BOX_ID_FIELD.into(), my_box_number);
    input.insert(
        FACES_DETECTED_FIELD.into(),
        HabValue::Integer(faces_detected as _),
    );
    input.insert(
        EXPECTED_MATCHES_FIELD.into(),
        HabValue::Integer(expected_matches),
    );
    // set box buffer field so that it is updated to the duplicated boxes if applicable. This should remove any possibility of discrepancies later
    input.insert(
        omz_utils::BOXES_BUFFER_FIELD.into(),
        HabValue::IntBuffer(duplicate_box_buffer),
    );

    input.insert("new_height_float".into(), new_height_float);
    input.insert("new_width_float".into(), new_width_float);
    input.insert("new_area_float".into(), new_area_float);
    input.insert("new_hw_ratio_float".into(), new_hw_ratio_float);

    std::mem::swap(&mut input, &mut outputs[0]);

    debug!(
        "split_bbs_before_scheduling created {} tuples with ids {:?}",
        outputs.len(),
        output_tuple_ids
    );

    outputs
}

pub fn preprocess_box_after_scheduling(
    mut input: Tuple,
    resize_width: u32,
    resize_height: u32,
    preprocessing: impl Fn(image::DynamicImage, u32, u32) -> ndarray::ArrayD<f32>,
) -> Vec<Tuple> {
    let Some(bb_val) = input.get(self::INDIVIDUAL_BOX_BOUND_FIELD) else {
        error!("bounding boxes field not found");
        return vec![];
    };
    let Some(bb) = bb_val.as_int_buffer() else {
        error!("bb field {INDIVIDUAL_BOX_BOUND_FIELD} is not an int buffer");
        return vec![];
    };
    let &[x1, y1, x2, y2] = bb else {
        error!(
            "bounding box in field {INDIVIDUAL_BOX_BOUND_FIELD} had {} elements (expected 4)",
            bb.len()
        );
        return vec![];
    };

    let Some(original_image_buffer) = input.get(ORIGINAL_IMAGE_FIELD) else {
        error!("original image field not found");
        return vec![];
    };
    let original_image_buffer = match original_image_buffer {
        HabValue::ByteBuffer(b) => b.as_slice(),
        HabValue::SharedArrayU8(b) => match b.0.as_slice() {
            Some(s) => s,
            None => {
                error!("original image field shared array is not contiguous");
                return vec![];
            }
        },
        v => {
            error!(
                "original image field not a byte buffer or shared array. its type was {:?}",
                v.get_type()
            );
            return vec![];
        }
    };
    let Some(original_image_shape) = input.get(ORIGINAL_IMAGE_SHAPE_FIELD) else {
        error!("original image shape field not found");
        return vec![];
    };
    let Some(original_image_shape) = original_image_shape.as_shape_buffer() else {
        error!("original image shape field not a shape buffer");
        return vec![];
    };
    let Some(image_id) = input.get(ORIGINAL_IMAGE_ID_FIELD) else {
        error!("original image id field not found");
        return vec![];
    };
    let Some(image_id) = image_id.as_string() else {
        error!("original image id field not a string");
        return vec![];
    };
    let Some(image_id_int) = input.get(ORIGINAL_IMAGE_ID_INT_FIELD) else {
        error!("original image id field not found");
        return vec![];
    };
    let Some(image_id_int) = image_id_int.as_integer() else {
        error!("original image id int field not an integer");
        return vec![];
    };

    let Ok(array_view) = ndarray::ArrayView3::<u8>::from_shape(
        (
            original_image_shape[0] as usize,
            original_image_shape[1] as usize,
            3,
        ),
        original_image_buffer,
    ) else {
        error!("failed to create array view from original image buffer. expected dimensions were {}x{}x3={}, but actual size was {}", original_image_shape[0], original_image_shape[1], 3, original_image_buffer.len());
        return vec![];
    };
    let array_view = array_view.slice(ndarray::s![
        (x1 as usize)..(x2 as usize),
        (y1 as usize)..(y2 as usize),
        ..
    ]);

    let Some(cropped_img) = image::RgbImage::from_raw(
        (x2 - x1) as u32,
        (y2 - y1) as u32,
        array_view.iter().copied().collect(),
    ) else {
        let w = x2 - x1;
        let h = y2 - y1;
        let expected = array_view.len();
        error!("failed to create cropped image from original image buffer. expected dimensions were {w}x{h}, but actual size was {array_view}");
        return vec![];
    };
    let cropped_img = image::DynamicImage::ImageRgb8(cropped_img);

    let processed_input = preprocessing(cropped_img, resize_width, resize_height);
    let processed_input_shape = processed_input.shape().to_owned();
    debug!("original image id {image_id} ({image_id_int}) created cropped image with shape {processed_input_shape:?}");

    let (processed_input, Some(0) | None) = processed_input.into_raw_vec_and_offset() else {
        error!("failed to get raw vec and offset");
        return vec![];
    };

    let cropped_image = HabValue::IntBuffer(bytemuck::cast_vec(processed_input));
    let cropped_image_shape = HabValue::ShapeBuffer(processed_input_shape.clone());
    let resized_shape = HabValue::ShapeBuffer(processed_input_shape);

    // remove any doubt that the borrow is still active
    // this used to be for an impl AsRef<[u8]>, but it has since been changed to a plain reference to a slice. it is harmless so I am keeping it in case it needs to come back later
    #[allow(dropping_references)]
    drop(original_image_buffer);

    input.insert("cropped_image".into(), cropped_image);
    input.insert("resized_shape".into(), resized_shape);
    input.insert("cropped_image_shape".into(), cropped_image_shape);

    let mut outputs = get_tuple_vec();
    outputs.push(input);
    outputs
}

pub fn preprocess_bb_facenet(input: Tuple) -> Vec<Tuple> {
    preprocess_box_after_scheduling(input, 160, 160, preprocess_facenet)
}

pub fn preprocess_bb_insightface(input: Tuple) -> Vec<Tuple> {
    preprocess_box_after_scheduling(input, 112, 112, preprocess_insightface)
}

pub fn resize(
    orig_image: &image::DynamicImage,
    new_width: u32,
    new_height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use fast_image_resize::IntoImageView;
    let Some(pixel_type) = orig_image.pixel_type() else {
        return Err("failed to get pixel type from original image".into());
    };
    let mut dst_image = fast_image_resize::images::Image::new(new_width, new_height, pixel_type);

    thread_local! {
        static RESIZER: std::cell::RefCell<fast_image_resize::Resizer> = std::cell::RefCell::new(fast_image_resize::Resizer::new());
    }
    RESIZER.with_borrow_mut(|resizer| {
        resizer.resize(
            orig_image,
            &mut dst_image,
            &Some(fast_image_resize::ResizeOptions {
                // algorithm: fast_image_resize::ResizeAlg::Convolution(
                //     fast_image_resize::FilterType::Box,
                // ),
                algorithm: fast_image_resize::ResizeAlg::Nearest,
                ..Default::default()
            }),
        )
    })?;
    let resized_img_buf = dst_image.into_vec();
    Ok(resized_img_buf)
}

pub fn resize_and_cast_u8(
    cropped_img: image::DynamicImage,
    new_width: u32,
    new_height: u32,
    preprocessing: impl Fn(ndarray::Array4<f32>) -> ndarray::ArrayD<f32>,
) -> ndarray::ArrayD<f32> {
    let save_image_steps = std::env::var("LOG_IMAGE_STEPS").is_ok();
    let resized_img_buf = match resize(&cropped_img, new_width, new_height) {
        Ok(v) => v,
        Err(e) => {
            error!("failed to resize image: {e}");
            return ndarray::Array3::zeros((0, 0, 0)).into_dyn();
        }
    };
    'save_image: {
        if save_image_steps {
            let resized_img = match image::load_from_memory(&resized_img_buf) {
                Ok(v) => v,
                Err(e) => {
                    error!("failed to load image from memory after crop+resize: {e}");
                    // return ndarray::Array3::zeros((0, 0, 0)).into_dyn();
                    break 'save_image;
                }
            };
            if let Err(e) = resized_img.save("resize_crop.jpg") {
                error!("failed to save image after crop+resize: {e:?}");
                break 'save_image;
            }
        }
    }

    use std::sync::LazyLock;
    // new static to handle env variable for setting HWC vs WHC
    const EMBED_ORDER_ENV_VAR: &str = "EMBED_IMAGE_ORDER_WHC";
    static EMBED_IMAGE_ORDER_HWC: LazyLock<bool> = LazyLock::new(|| {
        let Ok(mut setting) = std::env::var(EMBED_ORDER_ENV_VAR) else {
            debug!(
                "Environment variable {} not set, defaulting to HWC order",
                EMBED_ORDER_ENV_VAR
            );
            return true;
        };
        setting.make_ascii_lowercase();
        match setting.as_str().trim() {
            "1" | "true" | "yes" | "whc" => false,
            "0" | "false" | "no" | "hwc" => true,
            _ => {
                warn!("Unrecognized value for environment variable {}: {}. expected an HWC value ({}, {}, {}, {}) or a WHC value ({}, {}, {}, {}). Defaulting to HWC order.", EMBED_ORDER_ENV_VAR, setting, "1", "true", "yes", "hwc", "0", "false", "no", "whc");
                true
            }
        }
    });

    let resized_img = ndarray::Array3::<u8>::from_shape_vec(
        if *EMBED_IMAGE_ORDER_HWC {
            // order: HWC
            (new_height as usize, new_width as usize, 3)
        } else {
            // order: WHC
            // (new_width as usize, new_height as usize, 3)
            (new_width as usize, new_height as usize, 3)
        },
        resized_img_buf,
    );

    let mut use_hwc_order_dims = *EMBED_IMAGE_ORDER_HWC;
    const EMBED_ORDER_SWAP_ENV_VAR: &str = "EMBED_IMAGE_SWITCH_DIM_AXES";
    static EMBED_IMAGE_SWITCH_DIM_AXES: LazyLock<bool> = LazyLock::new(|| {
        let Ok(mut setting) = std::env::var(EMBED_ORDER_SWAP_ENV_VAR) else {
            debug!(
                "Environment variable {} not set, defaulting to not switching axes",
                EMBED_ORDER_SWAP_ENV_VAR
            );
            return false;
        };
        setting.make_ascii_lowercase();
        match setting.as_str().trim() {
            "1" | "true" | "yes" => true,
            "0" | "false" | "no" => false,
            _ => {
                warn!("Unrecognized value for environment variable {}: {}. expected a boolean value (1, true, yes) or (0, false, no). Defaulting to not switching axes.", EMBED_ORDER_SWAP_ENV_VAR, setting);
                false
            }
        }
    });

    if *EMBED_IMAGE_SWITCH_DIM_AXES {
        use_hwc_order_dims = !use_hwc_order_dims;
    }
    let resized_img = match resized_img {
        Ok(v) => v,
        Err(e) => {
            error!("failed to create array from shape vec: {e}");
            return ndarray::Array3::zeros((0, 0, 0)).into_dyn();
        }
    };

    let f32_resized_img = resized_img.mapv(|x| x as f32);
    let permutation_order = match (use_hwc_order_dims, *EMBED_IMAGE_SWITCH_DIM_AXES) {
        (true, false) => [2, 0, 1], // HWC -> CHW | (240, 320, 3) -> (3, 240, 320)
        (false, false) => [2, 1, 0], // WHC -> CHW | (320, 240, 3) -> (3, 240, 320)

        // We can tell these are likely to be incorrect from the way these dimensions are ordered
        // but we will leave them in for now in case they work out some reason
        (true, true) => [1, 0, 2], // HWC -> WHC | (320, 240, 3) -> (240, 320, 3)
        (false, true) => [0, 1, 2], // WHC -> HWC | (240, 320, 3) -> (240, 320, 3)
    };
    let transposed = f32_resized_img
        // .permuted_axes([2, 1, 0]) // WHC -> CHW
        // .permuted_axes([2, 0, 1]) // HWC -> CHW
        .permuted_axes(permutation_order)
        .iter()
        .copied()
        .collect::<Vec<_>>();

    let final_shape = (1, 3, new_height as usize, new_width as usize);
    let f32_resized_img =
        ndarray::Array4::from_shape_vec(final_shape, transposed).unwrap_or_else(|e| {
            error!("failed to transpose and reshape resized image: {e}");
            ndarray::Array4::zeros(final_shape)
        });
    preprocessing(f32_resized_img)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum FacenetNormalizationMethod {
    MinMax,
    Uniform,
}
// const FACENET_NORMALIZATION_METHOD: FacenetNormalizationMethod = FacenetNormalizationMethod::MinMax;
const FACENET_NORMALIZATION_METHOD: FacenetNormalizationMethod =
    FacenetNormalizationMethod::Uniform;

pub fn preprocess_facenet(
    cropped_img: image::DynamicImage,
    new_width: u32,
    new_height: u32,
) -> ndarray::ArrayD<f32> {
    let save_image_steps = std::env::var("LOG_IMAGE_STEPS").is_ok();
    // facenet is a normalized version of the image
    //    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    resize_and_cast_u8(cropped_img, new_width, new_height, |mut box_tensor| {
        match FACENET_NORMALIZATION_METHOD {
            FacenetNormalizationMethod::Uniform => {
                // this is how it's supposed to be normalized
                box_tensor = (box_tensor - 127.5) / 128.0;
            }

            FacenetNormalizationMethod::MinMax => {
                // this is how it is normalized in the April/May chroma index (with some added NaN handling)
                let max = box_tensor
                    .iter()
                    .copied()
                    .max_by(f32::total_cmp)
                    .unwrap_or(255.0)
                    .clamp(0.0, 255.0);
                let min = box_tensor
                    .iter()
                    .copied()
                    .min_by(f32::total_cmp)
                    .unwrap_or(0.0)
                    .clamp(0.0, 255.0);
                box_tensor = (box_tensor - min) / ((max - min) + 1e-6);
            }
        }
        if save_image_steps {
            debug!("resized+cropped array after facenet preprocessing:\n{box_tensor:?}");
        }
        box_tensor.into_dyn()
    })
}
// insightface is a noop
pub fn preprocess_insightface(
    cropped_img: image::DynamicImage,
    new_width: u32,
    new_height: u32,
) -> ndarray::ArrayD<f32> {
    let save_image_steps = std::env::var("LOG_IMAGE_STEPS").is_ok();
    resize_and_cast_u8(cropped_img, new_width, new_height, |mut box_tensor| {
        box_tensor = (box_tensor - 127.5) / 127.5;
        if save_image_steps {
            debug!("resized+cropped array after insightface preprocessing:\n{box_tensor:?}");
        }
        box_tensor.into_dyn()
    })
}

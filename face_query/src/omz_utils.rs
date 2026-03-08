use core::num;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use image::DynamicImage;
use ndarray::prelude::*;

use ndarray::s;
use watershed_shared::{basic_pooling::get_tuple_vec, HabValue, Tuple};

use crate::face_utils::{
    self, ORIGINAL_IMAGE_FIELD, ORIGINAL_IMAGE_ID_FIELD, ORIGINAL_IMAGE_SHAPE_FIELD,
};

const PROB_THRESHOLD: f32 = 0.7;
const IOU_THRESHOLD: f32 = 0.5;
const TOP_K: i32 = 200;

pub fn scale(bbox: &[i32; 4]) -> [i32; 4] {
    let width = bbox[2] - bbox[0];
    let height = bbox[3] - bbox[1];
    let maximum = width.max(height);
    let dx = ((maximum - width) / 2) as i32;
    let dy = ((maximum - height) / 2) as i32;

    [bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy]
}

pub fn crop_image(image: &DynamicImage, bbox: &[i32; 4]) -> DynamicImage {
    let cropped = image.crop_imm(
        bbox[0] as u32,
        bbox[1] as u32,
        (bbox[2] - bbox[0]) as u32,
        (bbox[3] - bbox[1]) as u32,
    );
    cropped
}

pub fn postprocess(
    score_buf: &[i32],
    score_dims: &[usize],
    box_buf: &[i32],
    box_dims: &[usize],
    original_width: i32,
    original_height: i32,
    prob_threshold: f32,
) -> (ndarray::Array2<i32>, Vec<i32>, ndarray::Array1<f32>) {
    let score_dims: &[usize; 2] = match score_dims.try_into() {
        Ok(v) => v,
        Err(_e) => {
            if score_dims.len() == 3 {
                if score_dims[0] != 1 {
                    error!("Score dims are not 2d AxB or 1xAxB, {:?}", score_dims);
                    return (
                        ndarray::Array2::zeros((0, 4)),
                        Vec::new(),
                        ndarray::Array1::zeros(0),
                    );
                }
                // otherwise it's fine to use the latter two
                score_dims[1..].try_into().unwrap()
            } else {
                error!("Score dims are not 2d AxB or 1xAxB, {:?}", score_dims);
                return (
                    ndarray::Array2::zeros((0, 4)),
                    Vec::new(),
                    ndarray::Array1::zeros(0),
                );
            }
        }
    };
    let scores = match ArrayView2::from_shape(*score_dims, bytemuck::cast_slice(score_buf)) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to reconstitute scores. We had a slice of length {} and we were asked to use a shape of {:?}. ndarray error: {:?}", score_buf.len(), score_dims, e);
            return (
                ndarray::Array2::zeros((0, 4)),
                Vec::new(),
                ndarray::Array1::zeros(0),
            );
        }
    };
    debug!("decoded scores with shape {:?}", scores.shape());
    // let scores = match scores
    //     .into_shape_with_order((4420, 2))
    //     {
    //         Ok(v) => v,
    //         Err(e) => {
    //             error!("Failed to reshape scores. We had a slice of length {} and we were asked to use a shape of {:?}. ndarray error: {:?}", score_buf.len(), score_dims, e);
    //             return (ndarray::Array2::zeros((0, 4)), Vec::new(), ndarray::Array1::zeros(0));
    //         }
    //     };

    // let boxes = outputs[1].view().into_shape_with_order((4420, 4)).expect(("Failed to reshape boxes"));
    let box_dims: &[usize; 2] = match box_dims.try_into() {
        Ok(v) => v,
        Err(_e) => {
            if box_dims.len() == 3 {
                if box_dims[0] != 1 {
                    error!("Box dims are not 2d AxB or 1xAxB, {:?}", box_dims);
                    return (
                        ndarray::Array2::zeros((0, 4)),
                        Vec::new(),
                        ndarray::Array1::zeros(0),
                    );
                }
                // otherwise it's fine to use the latter two
                box_dims[1..].try_into().unwrap()
            } else {
                error!("Box dims are not 2d AxB or 1xAxB, {:?}", box_dims);
                return (
                    ndarray::Array2::zeros((0, 4)),
                    Vec::new(),
                    ndarray::Array1::zeros(0),
                );
            }
        }
    };
    let boxes = match ArrayView2::from_shape(*box_dims, bytemuck::cast_slice(box_buf)) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to reconstitute boxes. We had a slice of length {} and we were asked to use a shape of {:?}. ndarray error: {:?}", box_buf.len(), box_dims, e);
            return (
                ndarray::Array2::zeros((0, 4)),
                Vec::new(),
                ndarray::Array1::zeros(0),
            );
        }
    };
    debug!("decoded boxes with shape {:?}", boxes.shape());
    // let boxes = boxes
    //     .into_shape_with_order((4420, 4))
    //     .expect(("Failed to reshape boxes"));
    let (boxes, labels, probs) = crate::omz_utils::predict(
        original_width as _,
        original_height as _,
        scores.view(),
        boxes.view(),
        prob_threshold,
        IOU_THRESHOLD, // 0.5,
        TOP_K,         // 200,
    );
    (boxes, labels, probs)
}

pub fn outputs_to_boxes(values: &[watershed_shared::HabValue]) -> HabValue {
    let Some(score_buf): Option<&[i32]> = values[0].as_int_buffer() else {
        let msg = "Failed to extract score buffer";
        error!("{}: {:?}", msg, values[0]);
        return HabValue::Null;
    };
    let Some(score_dims): Option<&[usize]> = values[1].as_shape_buffer() else {
        let msg = "Failed to extract score dims";
        error!("{}: {:?}", msg, values[0]);
        return HabValue::Null;
    };
    let Some(box_buf): Option<&[i32]> = values[2].as_int_buffer() else {
        let msg = "Failed to extract box buffer";
        error!("{}: {:?}", msg, values[0]);
        return HabValue::Null;
    };
    let Some(box_dims): Option<&[usize]> = values[3].as_shape_buffer() else {
        let msg = "Failed to extract box dims";
        error!("{}: {:?}", msg, values[0]);
        return HabValue::Null;
    };
    let Some(original_width): Option<i32> = values[4].into_integer() else {
        let msg = "Failed to extract width";
        error!("{}: {:?}", msg, values[0]);
        return HabValue::Null;
    };
    let Some(original_height): Option<i32> = values[5].into_integer() else {
        let msg = "Failed to extract height";
        error!("{}: {:?}", msg, values[0]);
        return HabValue::Null;
    };
    let Some(prob_threshold): Option<f32> = values[6].into_float().map(|v| v.0 as _) else {
        let msg = "Failed to extract prob_threshold";
        error!("{}: {:?}", msg, values[0]);
        return HabValue::Null;
    };
    let (boxes, _labels, _probs) = postprocess(
        score_buf,
        score_dims,
        box_buf,
        box_dims,
        original_width,
        original_height,
        prob_threshold,
    );

    let num_boxes = boxes.shape()[0];
    let (box_buf, extra) = boxes.into_raw_vec_and_offset();
    if let Some(extra @ 1..) = extra {
        let msg = "Failed to extract box buffer";
        error!("{}: {:?}", msg, extra);
        return HabValue::Null;
    }
    HabValue::IntBuffer(box_buf)
}

pub fn write_boxes_to_image(
    orig_image: &image::DynamicImage,
    boxes: &ndarray::Array2<i32>,
) -> image::DynamicImage {
    use imageproc::drawing::draw_hollow_rect_mut;
    use imageproc::rect::Rect as ImageRect;
    let color = image::Rgb([255u8, 128, 0]);
    let mut rgb_out = orig_image.to_rgb8();
    for i in 0..boxes.shape()[0] {
        let bbox = boxes.slice(s![i, ..]);
        let box_ = scale(&[bbox[0], bbox[1], bbox[2], bbox[3]]);
        println!("box_: {:?}", box_);
        // let cropped = crop_image(&orig_image, &box_);
        // cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)
        // cv2.imshow('', orig_image)
        let rect = ImageRect::at(box_[0], box_[1])
            .of_size((box_[2] - box_[0]) as u32, (box_[3] - box_[1]) as u32);
        draw_hollow_rect_mut(&mut rgb_out, rect, color);
        // draw_hollow_rect_mut(&mut orig_image.to_rgba8(), rect, color);
    }
    rgb_out
        .save("output_omz.png")
        .expect("Failed to save image");

    image::DynamicImage::ImageRgb8(rgb_out)
}

#[derive(Debug, Clone, Copy)]
enum StridedConversionMethod {
    MapVTogether = 0,
    FromFnWithIndexingTogether = 1,
    MapVCastThenCompute = 2,
    FromFnWithIndexingCastThenCompute = 3,
    ToOwnedBytesThenCastThenCompute = 4,

    // MapVTogetherSeparateManualMultiplyAdd=17,
    // FromFnWithIndexingTogetherSeparateManualMultiplyAdd=18,
    MapVCastThenComputeSeparateManualMultiplyAdd = 19,
    FromFnWithIndexingCastThenComputeSeparateManualMultiplyAdd = 20,
    ToOwnedBytesThenCastThenComputeSeparateManualMultiplyAdd = 21,

    // same as above, but checking to see if the explicit multiply+add is faster than relying on the compiler to optimize it out
    MapVTogetherManualMultiplyAdd = 33,
    FromFnWithIndexingTogetherManualMultiplyAdd = 34,
    MapVCastThenComputeManualMultiplyAdd = 35,
    FromFnWithIndexingCastThenComputeManualMultiplyAdd = 36,
    ToOwnedBytesThenCastThenComputeManualMultiplyAdd = 37,

    // Chunked versions to see if the compiler will automatically vectorize this
    MapVCastThenChunkedMultiplyAdd = 49,
    ToOwnedThenCastThenChunkedMultiplyAdd = 50,

    // SIMD versions
    MapVCastThenSimd = 65,
    ToOwnedThenCastThenSimd = 66,
}
const STRIDED_CONVERSION_METHOD_ARRAY: &[StridedConversionMethod] = &[
    StridedConversionMethod::MapVTogether,
    StridedConversionMethod::FromFnWithIndexingTogether,
    StridedConversionMethod::MapVCastThenCompute,
    StridedConversionMethod::FromFnWithIndexingCastThenCompute,
    StridedConversionMethod::ToOwnedBytesThenCastThenCompute,
    // StridedConversionMethod::MapVTogetherSeparateManualMultiplyAdd,
    // StridedConversionMethod::FromFnWithIndexingTogetherSeparateManualMultiplyAdd,
    StridedConversionMethod::MapVCastThenComputeSeparateManualMultiplyAdd,
    StridedConversionMethod::FromFnWithIndexingCastThenComputeSeparateManualMultiplyAdd,
    StridedConversionMethod::ToOwnedBytesThenCastThenComputeSeparateManualMultiplyAdd,
    StridedConversionMethod::MapVTogetherManualMultiplyAdd,
    StridedConversionMethod::FromFnWithIndexingTogetherManualMultiplyAdd,
    StridedConversionMethod::MapVCastThenComputeManualMultiplyAdd,
    StridedConversionMethod::FromFnWithIndexingCastThenComputeManualMultiplyAdd,
    StridedConversionMethod::ToOwnedBytesThenCastThenComputeManualMultiplyAdd,
    StridedConversionMethod::MapVCastThenChunkedMultiplyAdd,
    StridedConversionMethod::ToOwnedThenCastThenChunkedMultiplyAdd,
    StridedConversionMethod::MapVCastThenChunkedMultiplyAdd,
    StridedConversionMethod::ToOwnedThenCastThenChunkedMultiplyAdd,
    StridedConversionMethod::MapVCastThenSimd,
    StridedConversionMethod::ToOwnedThenCastThenSimd,
    StridedConversionMethod::MapVCastThenSimd,
    StridedConversionMethod::ToOwnedThenCastThenSimd,
];

#[derive(Debug)]
struct StridedConversionMethodOptions {
    should_rotate: std::sync::atomic::AtomicBool,
    current_index: std::sync::atomic::AtomicUsize,
}
const ROTATING: StridedConversionMethodOptions = StridedConversionMethodOptions {
    should_rotate: std::sync::atomic::AtomicBool::new(true),
    current_index: std::sync::atomic::AtomicUsize::new(0),
};
static STRIDED_CONVERSION_METHOD_OPTION: std::sync::LazyLock<StridedConversionMethodOptions> =
    std::sync::LazyLock::new(|| {
        let Ok(mut v) = std::env::var("OMZ_STRIDED_CONVERSION_METHOD_STATIC") else {
            return ROTATING;
        };
        v.make_ascii_lowercase();
        let v = v.trim();
        let Ok(v) = v.parse::<usize>() else {
            warn!("Unrecognized value for environment variable OMZ_STRIDED_CONVERSION_METHOD_STATIC: {}. expected an integer in from the following: {:?}. Defaulting to Rotate", v, STRIDED_CONVERSION_METHOD_ARRAY.iter().map(|v| *v as usize).collect::<Vec<_>>());
            return ROTATING;
        };
        if !STRIDED_CONVERSION_METHOD_ARRAY
            .iter()
            .any(|&x| x as usize == v)
        {
            warn!("Unrecognized value for environment variable OMZ_STRIDED_CONVERSION_METHOD_STATIC: {}. expected an integer in from the following: {:?}. Defaulting to Rotate", v, STRIDED_CONVERSION_METHOD_ARRAY.iter().map(|v| *v as usize).collect::<Vec<_>>());
            return ROTATING;
        }
        info!("Using static strided conversion method {} as specified by environment variable OMZ_STRIDED_CONVERSION_METHOD_STATIC", v);
        StridedConversionMethodOptions {
            should_rotate: std::sync::atomic::AtomicBool::new(false),
            current_index: std::sync::atomic::AtomicUsize::new(v),
        }
    });

const PIXEL_TYPE_VAL: fast_image_resize::PixelType = fast_image_resize::PixelType::U8x3;
type PixelType = fast_image_resize::pixels::U8x3;
// pub fn preprocess(orig_image: &image::DynamicImage) -> ndarray::Array4<f32> {
// pub fn preprocess(orig_image: &image::SubImage<image::Rgb<u8>>) -> ndarray::Array4<f32> {
pub fn preprocess(orig_image: &[u8], old_width: u32, old_height: u32) -> ndarray::Array4<f32> {
    let extract_start = std::time::Instant::now();
    // let orig_image : &[PixelType] = bytemuck::cast_slice(orig_image);
    let orig_img_reinterpret_start = std::time::Instant::now();
    let orig_image: &[PixelType] = if orig_image.len() % 3 == 0 {
        // SAFETY: we checked that the length is a multiple of 3, so it must be composed of pixels
        unsafe { std::mem::transmute(orig_image) }
    } else {
        error!("Image length is not a multiple of 3, cannot cast to PixelType RGB");
        return ndarray::Array4::zeros((1, 3, 240, 320));
    };
    let orig_img_reinterpret_elapsed_micros = orig_img_reinterpret_start.elapsed().as_micros();

    let imgref_start = std::time::Instant::now();
    let Ok(orig_image) =
        fast_image_resize::images::ImageRef::from_pixels(old_width, old_height, orig_image)
    else {
        error!("failed to create ImageRef from original image");
        return ndarray::Array4::zeros((1, 3, 240, 320));
    };
    let imgref_elapsed_micros = imgref_start.elapsed().as_micros();

    let new_width = 320;
    let new_height = 240;
    // use fast_image_resize::IntoImageView;
    // let Some(pixel_type) = orig_image.pixel_type() else {
    //     error!("failed to get pixel type from cropped image");
    //     return ndarray::Array4::zeros((0, 0, 0, 0));
    // };
    // let mut dst_image = fast_image_resize::images::Image::new(new_width, new_height, pixel_type);
    let get_resize_config_start = std::time::Instant::now();
    static OMZ_USE_NEAREST_RESIZE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("OMZ_USE_NEAREST_RESIZE").is_ok());

    let algorithm = if *OMZ_USE_NEAREST_RESIZE {
        fast_image_resize::ResizeAlg::Nearest
    } else {
        fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Box)
    };
    let get_resize_config_elapsed_micros = get_resize_config_start.elapsed().as_micros();

    let alloc_dst_image_start = std::time::Instant::now();
    let mut dst_image =
        fast_image_resize::images::Image::new(new_width, new_height, PIXEL_TYPE_VAL);
    let alloc_dst_image_elapsed_micros = alloc_dst_image_start.elapsed().as_micros();

    use std::cell::RefCell;
    thread_local! {
        static RESIZER: RefCell<fast_image_resize::Resizer> = RefCell::new(fast_image_resize::Resizer::new());
    }
    // static RESIZER: LazyLock<std::sync::Mutex<fast_image_resize::Resizer>> =
    //     LazyLock::new(|| std::sync::Mutex::new(fast_image_resize::Resizer::new()));
    // let mut resizer = if let Ok(guard) = RESIZER.lock() {
    //     guard
    // } else {
    //     error!("failed to lock resizer mutex");
    //     return ndarray::Array4::zeros((1, 3, 240, 320));
    // };

    let lock_aquire_start = std::time::Instant::now();
    let resize_start = std::time::Instant::now();
    let mut lock_aquire_elapsed_micros = 0;
    let resize_result = RESIZER.with(|r| {
        let mut r = r.borrow_mut();
        lock_aquire_elapsed_micros = lock_aquire_start.elapsed().as_micros();
        r.resize(
            &orig_image,
            &mut dst_image,
            &Some(fast_image_resize::ResizeOptions {
                algorithm,
                ..Default::default()
            }),
        )
    });
    if let Err(e) = resize_result {
        error!("failed to resize image using fast_image_resize: {e}");
        return ndarray::Array4::zeros((1, 3, 240, 320));
    }
    let resize_elapsed_micros = resize_start.elapsed().as_micros();
    let extract_to_resize_total_micros = extract_start.elapsed().as_micros();
    debug!("Early component timings for detect preprocess (extract+resize total: {extract_to_resize_total_micros} us): reinterpret: {orig_img_reinterpret_elapsed_micros} us, imgref: {imgref_elapsed_micros} us, get_resize_config: {get_resize_config_elapsed_micros} us, alloc_dst_image: {alloc_dst_image_elapsed_micros} us, lock_aquire: {lock_aquire_elapsed_micros} us, resize: {resize_elapsed_micros} us");

    let resized_img_buf: Vec<u8> = dst_image.into_vec();
    // let resized_img_buf = match crate::face_utils::resize(&orig_image, new_width, new_height) {
    //     Ok(v) => v,
    //     Err(e) => {
    //         error!("failed to resize image using crate::face_utils::resize: {e}");
    //         return ndarray::Array4::zeros((0, 0, 0, 0));
    //     }
    // };

    use std::sync::LazyLock;
    // new static to handle env variable for setting HWC vs WHC
    const DETECT_ORDER_ENV_VAR: &str = "DETECT_IMAGE_ORDER_WHC";
    static DETECT_IMAGE_ORDER_HWC: LazyLock<bool> = LazyLock::new(|| {
        let Ok(mut setting) = std::env::var(DETECT_ORDER_ENV_VAR) else {
            debug!(
                "Environment variable {} not set, defaulting to HWC order",
                DETECT_ORDER_ENV_VAR
            );
            return true;
        };
        setting.make_ascii_lowercase();
        match setting.as_str().trim() {
            "1" | "true" | "yes" | "whc" => false,
            "0" | "false" | "no" | "hwc" => true,
            _ => {
                warn!("Unrecognized value for environment variable {}: {}. expected an HWC value ({}, {}, {}, {}) or a WHC value ({}, {}, {}, {}). Defaulting to HWC order.", DETECT_ORDER_ENV_VAR, setting, "1", "true", "yes", "hwc", "0", "false", "no", "whc");
                true
            }
        }
    });

    let shape = if *DETECT_IMAGE_ORDER_HWC {
        // order: HWC
        (new_height as usize, new_width as usize, 3)
    } else {
        // order: WHC
        (new_width as usize, new_height as usize, 3)
    };
    let resized_img = ndarray::Array3::<u8>::from_shape_vec(shape, resized_img_buf);
    let mut use_hwc_order_dims = *DETECT_IMAGE_ORDER_HWC;
    const DETECT_ORDER_SWAP_ENV_VAR: &str = "DETECT_IMAGE_SWITCH_DIM_AXES";
    static DETECT_IMAGE_ORDER_SWAP: LazyLock<bool> = LazyLock::new(|| {
        let Ok(mut setting) = std::env::var(DETECT_ORDER_SWAP_ENV_VAR) else {
            debug!(
                "Environment variable {} not set, defaulting to not switching axes",
                DETECT_ORDER_SWAP_ENV_VAR
            );
            return false;
        };
        setting.make_ascii_lowercase();
        match setting.as_str().trim() {
            "1" | "true" | "yes" => true,
            "0" | "false" | "no" => false,
            _ => {
                warn!("Unrecognized value for environment variable {}: {}. expected a boolean value (1, true, yes) or (0, false, no). Defaulting to not switching axes.", DETECT_ORDER_SWAP_ENV_VAR, setting);
                false
            }
        }
    });

    if *DETECT_IMAGE_ORDER_SWAP {
        use_hwc_order_dims = !use_hwc_order_dims;
    }

    let mut arr_image: ArrayBase<ndarray::OwnedRepr<u8>, Dim<[usize; 3]>> = match resized_img {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to reshape image to (240, 320, 3): {e}");
            return ndarray::Array4::zeros((1, 3, 240, 320));
        }
    };

    let dimension_start = std::time::Instant::now();
    // regular constants to add then divide
    const IMAGE_ADD: f32 = -127.0;
    const IMAGE_DIV: f32 = 128.0;
    // manual constants to multiply then add
    const IMAGE_MULTIPLY: f32 = 1.0 / 128.0;
    const IMAGE_ADD_MANUAL: f32 = -127.0 * IMAGE_MULTIPLY;

    if use_hwc_order_dims {
        // supposing we were in HWC order, we need to go to CHW order
        // we start with (240, 320, 3) and want to get to (3, 240, 320)
        arr_image.swap_axes(0, 2); // (240, 320, 3) -> (3, 320, 240)
        arr_image.swap_axes(1, 2); // (3, 320, 240) -> (3, 240, 320)
    } else {
        // supposing we were in WHC order, we need to go to CHW order
        // we start with (320, 240, 3) and want to get to (3, 240, 320)
        arr_image.swap_axes(0, 2); // (320, 240, 3) -> (3, 240, 320)
    }
    let arr_image = arr_image.insert_axis(ndarray::Axis(0));

    let correction_start = std::time::Instant::now();
    let dims: (usize, usize, usize, usize) = arr_image.dim();
    let should_rotate = STRIDED_CONVERSION_METHOD_OPTION
        .should_rotate
        .load(std::sync::atomic::Ordering::Relaxed);
    let strided_conversion_method = if should_rotate {
        let current_index = STRIDED_CONVERSION_METHOD_OPTION
            .current_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % STRIDED_CONVERSION_METHOD_ARRAY.len();
        STRIDED_CONVERSION_METHOD_ARRAY[current_index]
    } else {
        let current_index = STRIDED_CONVERSION_METHOD_OPTION
            .current_index
            .load(std::sync::atomic::Ordering::Relaxed);
        if let Some(&v) = STRIDED_CONVERSION_METHOD_ARRAY
            .iter()
            .find(|&&v| v as usize == current_index)
        {
            v
        } else {
            error!("Unrecognized static strided conversion method index {}, defaulting to MapVTogether and fixing future iterations to use that.", current_index);
            STRIDED_CONVERSION_METHOD_OPTION.current_index.store(
                StridedConversionMethod::MapVTogether as usize,
                std::sync::atomic::Ordering::Relaxed,
            );
            StridedConversionMethod::MapVTogether
        }
    };
    let strided_conversion_method_int = strided_conversion_method as usize;
    let mut intermediate_times_micros = smallvec::SmallVec::<[f64; 8]>::new();
    let mut log_split_time =
        || intermediate_times_micros.push(correction_start.elapsed().as_nanos() as f64 / 1000.0);

    let arr_image = match strided_conversion_method {
        StridedConversionMethod::MapVTogether => {
            arr_image.mapv(|x| (x as f32 + IMAGE_ADD) / IMAGE_DIV)
        }
        StridedConversionMethod::MapVTogetherManualMultiplyAdd => {
            arr_image.mapv(|x| ((x as f32) * IMAGE_MULTIPLY) + IMAGE_ADD_MANUAL)
        }

        StridedConversionMethod::FromFnWithIndexingTogether => {
            ndarray::Array4::from_shape_fn(dims, |coord| {
                (arr_image[coord] as f32 + IMAGE_ADD) / IMAGE_DIV
            })
        }
        StridedConversionMethod::FromFnWithIndexingTogetherManualMultiplyAdd => {
            ndarray::Array4::from_shape_fn(dims, |coord| {
                ((arr_image[coord] as f32) * IMAGE_MULTIPLY) + IMAGE_ADD_MANUAL
            })
        }


        StridedConversionMethod::MapVCastThenCompute => {
            let mut a = arr_image.mapv(|x| x as f32);
            log_split_time();
            a.mapv_inplace(|x| (x + IMAGE_ADD) / IMAGE_DIV);
            a
        }
        StridedConversionMethod::MapVCastThenComputeManualMultiplyAdd => {
            let mut a = arr_image.mapv(|x| x as f32);
            log_split_time();
            a.mapv_inplace(|x| (x * IMAGE_MULTIPLY) + IMAGE_ADD_MANUAL);
            a
        }
        StridedConversionMethod::MapVCastThenComputeSeparateManualMultiplyAdd => {
            let mut a = arr_image.mapv(|x| x as f32);
            log_split_time();
            a.mapv_inplace(|x| (x * IMAGE_MULTIPLY));
            log_split_time();
            a.mapv_inplace(|v| v + IMAGE_ADD_MANUAL);
            a
        }

        StridedConversionMethod::FromFnWithIndexingCastThenCompute => {
            let mut a = ndarray::Array4::from_shape_fn(dims, |coord| arr_image[coord] as f32);
            log_split_time();
            a.mapv_inplace(|x| (x + IMAGE_ADD) / IMAGE_DIV);
            a
        }
        StridedConversionMethod::FromFnWithIndexingCastThenComputeManualMultiplyAdd => {
            let mut a = ndarray::Array4::from_shape_fn(dims, |coord| arr_image[coord] as f32);
            log_split_time();
            a.mapv_inplace(|x| (x * IMAGE_MULTIPLY) + IMAGE_ADD_MANUAL);
            a
        }
        StridedConversionMethod::FromFnWithIndexingCastThenComputeSeparateManualMultiplyAdd => {
            let mut a = ndarray::Array4::from_shape_fn(dims, |coord| arr_image[coord] as f32);
            log_split_time();
            a.mapv_inplace(|x| (x * IMAGE_MULTIPLY));
            log_split_time();
            a.mapv_inplace(|v| v + IMAGE_ADD_MANUAL);
            a
        }

        method @ StridedConversionMethod::ToOwnedBytesThenCastThenCompute => {
            let rearranged_bytes: Vec<u8> = arr_image.iter().copied().collect();
            log_split_time();
            let rearranged_as_floats = rearranged_bytes.into_iter().map(|x| x as f32).collect();
            log_split_time();
            let mut a = match ndarray::Array4::from_shape_vec(dims, rearranged_as_floats) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to reshape image to (1, 3, 240, 320) in {method:?} method: {e}");
                    return ndarray::Array4::zeros((1, 3, 240, 320));
                }
            };
            a.mapv_inplace(|x| (x + IMAGE_ADD) / IMAGE_DIV);
            a
        }
        method @ StridedConversionMethod::ToOwnedBytesThenCastThenComputeManualMultiplyAdd => {
            let rearranged_bytes: Vec<u8> = arr_image.iter().copied().collect();
            log_split_time();
            let rearranged_as_floats = rearranged_bytes.into_iter().map(|x| x as f32).collect();
            log_split_time();
            let mut a = match ndarray::Array4::from_shape_vec(dims, rearranged_as_floats) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to reshape image to (1, 3, 240, 320) in {method:?} method: {e}");
                    return ndarray::Array4::zeros((1, 3, 240, 320));
                }
            };
            a.mapv_inplace(|x| (x * IMAGE_MULTIPLY) + IMAGE_ADD_MANUAL);
            a
        }
        method @ StridedConversionMethod::ToOwnedBytesThenCastThenComputeSeparateManualMultiplyAdd => {
            let rearranged_bytes: Vec<u8> = arr_image.iter().copied().collect();
            log_split_time();
            let rearranged_as_floats = rearranged_bytes.into_iter().map(|x| x as f32).collect();
            log_split_time();
            let mut a = match ndarray::Array4::from_shape_vec(dims, rearranged_as_floats) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to reshape image to (1, 3, 240, 320) in {method:?} method: {e}");
                    return ndarray::Array4::zeros((1, 3, 240, 320));
                }
            };
            a.mapv_inplace(|x| x * IMAGE_MULTIPLY);
            log_split_time();
            a.mapv_inplace(|v| v + IMAGE_ADD_MANUAL);
            a
        }
        method @ StridedConversionMethod::MapVCastThenChunkedMultiplyAdd => {
            let mut a = arr_image.mapv(|x| x as f32);
            log_split_time();
            const CHUNK_SIZE: usize = 8;
            if let Err(a) = chunked_multiply_add::<CHUNK_SIZE>(&mut a, method) {
                return a;
            }
            a
        }
        method @ StridedConversionMethod::ToOwnedThenCastThenChunkedMultiplyAdd => {
            let rearranged_bytes: Vec<u8> = arr_image.iter().copied().collect();
            log_split_time();
            let rearranged_as_floats = rearranged_bytes.into_iter().map(|x| x as f32).collect();
            log_split_time();
            let mut a = match ndarray::Array4::from_shape_vec(dims, rearranged_as_floats) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to reshape image to (1, 3, 240, 320) in {method:?} method: {e}");
                    return ndarray::Array4::zeros((1, 3, 240, 320));
                }
            };
            const CHUNK_SIZE: usize = 8;
            if let Err(a) = chunked_multiply_add::<CHUNK_SIZE>(&mut a, method) {
                return a;
            }
            a
        }
        method @ StridedConversionMethod::MapVCastThenSimd => {
            let mut a = arr_image.mapv(|x| x as f32);
            log_split_time();
            if let Err(a) = chunked_simd_multiply_add(&mut a, method) {
                return a;
            }
            a
        }
        method @ StridedConversionMethod::ToOwnedThenCastThenSimd => {
            let rearranged_bytes: Vec<u8> = arr_image.iter().copied().collect();
            log_split_time();
            let rearranged_as_floats = rearranged_bytes.into_iter().map(|x| x as f32).collect();
            log_split_time();
            let mut a = match ndarray::Array4::from_shape_vec(dims, rearranged_as_floats) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to reshape image to (1, 3, 240, 320) in {method:?} method: {e}");
                    return ndarray::Array4::zeros((1, 3, 240, 320));
                }
            };
            if let Err(a) = chunked_simd_multiply_add(&mut a, method) {
                return a;
            }
            a
        }
    };

    fn chunked_multiply_add<const CHUNK_SIZE: usize>(
        a: &mut ndarray::Array4<f32>,
        method: StridedConversionMethod,
    ) -> Result<(), ndarray::Array4<f32>> {
        let total_size = a.len();
        let chunks = total_size / CHUNK_SIZE;
        let remainder = total_size % CHUNK_SIZE;
        let Some(a_slice) = a.as_slice_mut() else {
            error!("Expected array to be contiguous when using method {method:?} but it had strides {:?}", a.strides());
            return Err(ndarray::Array4::zeros((1, 3, 240, 320)));
        };
        for chunk in 0..chunks {
            let start = chunk * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;
            let chunk_slice = &mut a_slice[start..end];
            for v in chunk_slice.iter_mut() {
                *v = (*v * IMAGE_MULTIPLY) + IMAGE_ADD_MANUAL;
            }
        }
        if remainder > 0 {
            let start = chunks * CHUNK_SIZE;
            let chunk_slice = &mut a_slice[start..];
            for v in chunk_slice.iter_mut() {
                *v = (*v * IMAGE_MULTIPLY) + IMAGE_ADD_MANUAL;
            }
        }
        Ok(())
    }
    // we will do f32x8 SIMD operations
    const SIMD_CHUNK_SIZE: usize = 8;
    type SimdType = wide::f32x8;
    // assert length is correct
    const _: () = {
        assert!(std::mem::size_of::<SimdType>() == SIMD_CHUNK_SIZE * std::mem::size_of::<f32>());
    };
    fn chunked_simd_multiply_add(
        a: &mut ndarray::Array4<f32>,
        method: StridedConversionMethod,
    ) -> Result<(), ndarray::Array4<f32>> {
        let total_size = a.len();
        let chunks = total_size / SIMD_CHUNK_SIZE;
        let remainder = total_size % SIMD_CHUNK_SIZE;
        let Some(a_slice) = a.as_slice_mut() else {
            error!("Expected array to be contiguous when using method {method:?} but it had strides {:?}", a.strides());
            return Err(ndarray::Array4::zeros((1, 3, 240, 320)));
        };
        for chunk in 0..chunks {
            let start = chunk * SIMD_CHUNK_SIZE;
            let end = start + SIMD_CHUNK_SIZE;
            let chunk_slice = &mut a_slice[start..end];
            let chunk_arr: &mut [f32; SIMD_CHUNK_SIZE] = TryFrom::try_from(chunk_slice)
                .expect("already chunked with slice with correct length");
            let simd_vals = SimdType::from(*chunk_arr);
            const IMAGE_MULTIPLY_SIMD: SimdType = SimdType::splat(IMAGE_MULTIPLY);
            const IMAGE_ADD_SIMD: SimdType = SimdType::splat(IMAGE_ADD_MANUAL);
            let simd_outputs = simd_vals.mul_add(IMAGE_MULTIPLY_SIMD, IMAGE_ADD_SIMD);
            *chunk_arr = simd_outputs.into();
        }
        if remainder > 0 {
            let start = chunks * SIMD_CHUNK_SIZE;
            let chunk_slice = &mut a_slice[start..];
            for v in chunk_slice.iter_mut() {
                *v = (*v * IMAGE_MULTIPLY) + IMAGE_ADD_MANUAL;
            }
        }
        Ok(())
    }

    log_split_time();
    let correction_elapsed_micros = correction_start.elapsed().as_nanos() as f64 / 1000.0;
    debug!("conversion method #{strided_conversion_method_int} ({strided_conversion_method:?}) completed in {correction_elapsed_micros} us, intermediate diffs were at the following times: {intermediate_times_micros:?}");

    // let arr_image = ndarray::Array4::from_shape_fn(dims, |coord| {
    //     (arr_image[coord] as f32 + IMAGE_ADD) / IMAGE_DIV
    // });

    // let arr_image = arr_image.mapv(|x| (x as f32 + IMAGE_ADD) / IMAGE_DIV);

    // let mut arr_image = arr_image.mapv(|x| (x as f32 + IMAGE_ADD) / IMAGE_DIV);

    // let mut arr_image = arr_image.mapv(|x| x as f32);
    // let image_mean = [127, 127, 127];
    // for channel in 0..3 {
    //     let mut arr_image = arr_image.slice_mut(s![.., .., channel]);
    //     // arr_image.mapv_inplace(|x| x as f32);
    //     // arr_image.mapv_inplace(|x| x - image_mean[channel] as f32);
    //     arr_image -= image_mean[channel] as f32;
    // }
    // // arr_image.mapv_inplace(
    // arr_image /= 128.0;
    // let image = arr_image.(2, 0, 1);
    // transpose the array to (1, 3, 240, 320)

    // let dimension_start = std::time::Instant::now();
    // arr_image.swap_axes(0, 2);
    // arr_image.swap_axes(1, 2);
    // // let image = arr_image.insert_axis(ndarray::Axis(0));
    // let mut dims: [usize; 3] = [0; 3];
    // for d in 0..3 {
    //     dims[d] = arr_image.shape()[d];
    // }
    // let reconstruct_start = std::time::Instant::now();

    // reconstruct it as owned
    // let output_len: usize = dims.iter().copied().product();
    // let mut output_arr = Vec::with_capacity(output_len);
    // output_arr.extend(arr_image.into_iter());
    // let Ok(arr_image) = ndarray::Array3::from_shape_vec(dims, output_arr)
    // else {
    //     error!("Failed to reshape image to (1, 3, 240, 320)");
    //     return ndarray::Array4::zeros((1, 3, 240, 320));
    // };
    // let image = arr_image.insert_axis(ndarray::Axis(0));

    // let image = arr_image.insert_axis(ndarray::Axis(0)).into_owned();

    let image = arr_image;

    let preprocess_end = std::time::Instant::now();
    let timing_extract = resize_start.duration_since(extract_start).as_micros();
    let timing_resize = dimension_start.duration_since(resize_start).as_micros();
    let timing_dimension = correction_start.duration_since(dimension_start).as_micros();
    let timing_corrections = preprocess_end.duration_since(correction_start).as_micros();
    let timing_total = preprocess_end.duration_since(extract_start).as_micros();
    debug!(
        "timing: extraction: {:?} us, resizing: {:?} us, dimension swaps: {:?} us, correction computation: {:?} us, total: {:?} us",
        timing_extract, timing_resize, timing_dimension, timing_corrections, timing_total
    );
    image
}

pub fn area_of(
    left_top: ndarray::ArrayView2<'_, f32>,
    right_bottom: ndarray::ArrayView2<'_, f32>,
) -> ndarray::Array1<f32> {
    let hw = (right_bottom.to_owned() - left_top).mapv(|x| x.max(0.0));
    hw.slice(s![.., 0]).to_owned() * hw.slice(s![.., 1]).to_owned()
}

pub fn iou_of(
    boxes0: &ndarray::Array2<f32>,
    boxes1: &ndarray::Array2<f32>,
    eps: f32,
) -> ndarray::Array1<f32> {
    assert_eq!(boxes0.shape()[1], 4);
    assert_eq!(boxes1.shape()[1], 4);
    let mut overlap_left_top = boxes0.slice(s![.., ..2]).to_owned();
    for box0_number in 0..boxes0.shape()[0] {
        for box1_number in 0..boxes1.shape()[0] {
            overlap_left_top[[box0_number, 0]] =
                overlap_left_top[[box0_number, 0]].max(boxes1[[box1_number, 0]]);
            overlap_left_top[[box0_number, 1]] =
                overlap_left_top[[box0_number, 1]].max(boxes1[[box1_number, 1]]);
        }
    }
    let mut overlap_right_bottom = boxes0.slice(s![.., 2..]).to_owned();
    for box0_number in 0..boxes0.shape()[0] {
        for box1_number in 0..boxes1.shape()[0] {
            overlap_right_bottom[[box0_number, 0]] =
                overlap_right_bottom[[box0_number, 0]].min(boxes1[[box1_number, 2]]);
            overlap_right_bottom[[box0_number, 1]] =
                overlap_right_bottom[[box0_number, 1]].min(boxes1[[box1_number, 3]]);
        }
    }
    let overlap_area = area_of(overlap_left_top.view(), overlap_right_bottom.view());
    let area0 = area_of(boxes0.slice(s![.., ..2]), boxes0.slice(s![.., 2..]));
    let area1 = area_of(boxes1.slice(s![.., ..2]), boxes1.slice(s![.., 2..]));
    let divisor = overlap_area.to_owned();
    let iou = divisor / (area0 + area1 - overlap_area + eps);
    iou
}

pub fn hard_nms(
    box_scores: &ndarray::Array2<f32>,
    iou_threshold: f32,
    top_k: i32,
    candidate_size: usize,
) -> ndarray::Array2<f32> {
    let scores = box_scores.slice(s![.., -1]).to_owned();
    let boxes = box_scores.slice(s![.., ..-1]).to_owned();
    let mut picked = Vec::new();
    let mut indexes = scores.iter().enumerate().collect::<Vec<_>>();
    indexes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    indexes.truncate(candidate_size);
    while !indexes.is_empty() {
        let current = indexes.last().unwrap().0;
        picked.push(current);
        if (top_k > 0 && picked.len() as i32 == top_k) || indexes.len() == 1 {
            break;
        }
        let current_box = boxes.slice(s![current, ..]);
        indexes.pop();
        let rest_boxes = boxes.select(
            ndarray::Axis(0),
            &indexes.iter().map(|x| x.0).collect::<Vec<_>>(),
        );
        let iou = iou_of(
            &rest_boxes,
            &current_box.to_owned().insert_axis(ndarray::Axis(0)),
            1e-5,
        );
        // zip with mask
        let mut iou_mask_iter = iou.iter().map(|&x| x <= iou_threshold);
        // indexes.retain(|&x| iou[x.0] <= iou_threshold);
        indexes.retain(|&_x| iou_mask_iter.next().unwrap_or_default());
    }
    box_scores.select(ndarray::Axis(0), &picked)
}

pub fn predict(
    width: i32,
    height: i32,
    confidences: ndarray::ArrayView2<'_, f32>,
    boxes: ndarray::ArrayView2<'_, f32>,
    prob_threshold: f32,
    iou_threshold: f32,
    top_k: i32,
) -> (ndarray::Array2<i32>, Vec<i32>, ndarray::Array1<f32>) {
    let mut picked_box_probs = Vec::new();
    let mut picked_labels = Vec::new();
    for class_index in 1..confidences.shape()[1] {
        let probs = confidences.slice(s![.., class_index]);
        let mask = probs
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| (x > prob_threshold).then(|| i))
            .collect::<Vec<_>>();
        let probs = probs.select(ndarray::Axis(0), &mask);
        if probs.shape()[0] == 0 {
            continue;
        }
        let subset_boxes = boxes.select(ndarray::Axis(0), &mask);
        println!("subset_boxes: {:?}", subset_boxes.shape());
        println!("probs: {:?}", probs.shape());
        let mut box_probs = ndarray::concatenate(
            ndarray::Axis(1),
            &[
                subset_boxes.view(),
                probs.view().insert_axis(ndarray::Axis(1)),
            ],
        )
        .unwrap();
        box_probs = hard_nms(&box_probs, iou_threshold, top_k, 200);
        picked_labels.extend(vec![class_index as i32; box_probs.shape()[0]]);
        picked_box_probs.push(box_probs);
    }
    if picked_box_probs.is_empty() {
        return (
            ndarray::Array2::<i32>::zeros((0, 4)),
            Vec::new(),
            ndarray::Array1::<f32>::zeros(0),
        );
    }
    let picked_box_probs = ndarray::concatenate(
        ndarray::Axis(0),
        &picked_box_probs
            .iter()
            .map(|v| v.view())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut picked_box_probs = picked_box_probs.to_owned();
    picked_box_probs
        .slice_mut(s![.., 0])
        .mapv_inplace(|x| x * width as f32);
    picked_box_probs
        .slice_mut(s![.., 1])
        .mapv_inplace(|x| x * height as f32);
    picked_box_probs
        .slice_mut(s![.., 2])
        .mapv_inplace(|x| x * width as f32);
    picked_box_probs
        .slice_mut(s![.., 3])
        .mapv_inplace(|x| x * height as f32);
    let picked_probs = picked_box_probs.slice(s![.., 4]).to_owned();
    let picked_box_probs = picked_box_probs.slice(s![.., ..4]).mapv(|x| x as i32);
    let picked_labels = picked_labels.into_iter().map(|x| x as i32).collect();
    (picked_box_probs, picked_labels, picked_probs)
}

const DETECT_INPUT_FIELD: &str = "detect_input";
const DETECT_SHAPE_FIELD: &str = "detect_shape";

pub fn preprocess_image(mut t: Tuple) -> Vec<Tuple> {
    let mut v = get_tuple_vec();

    let Some(original_image) = t.get(ORIGINAL_IMAGE_FIELD) else {
        error!("Failed to extract original image");
        return v;
    };

    let Some(original_image) = original_image.as_shared_array_u8() else {
        error!("Failed to extract original image");
        return v;
    };
    let original_image_shape = original_image.shape();
    let Some(original_image_buf) = original_image.as_slice() else {
        error!("original image buffer was not contiguous");
        return v;
    };

    let width = original_image_shape[0] as u32;
    let height = original_image_shape[1] as u32;

    let start_time = std::time::Instant::now();
    let preprocessed_image = preprocess(original_image_buf, width, height);
    let img_proc_micros = start_time.elapsed().as_micros();
    debug!(
        "preprocessing of image for tuple {} (id={:?}, w={:?}, h={:?}) took {:?} micros",
        t.id(),
        t.get(ORIGINAL_IMAGE_ID_FIELD),
        width,
        height,
        img_proc_micros
    );
    let preprocessed_image_shape = preprocessed_image.shape().to_vec();
    let (preprocessed_image_buf, extra) = preprocessed_image.into_raw_vec_and_offset();
    if let Some(extra @ 1..) = extra {
        error!("Failed to extract preprocessed image buffer: {:?}", extra);
        return v;
    }
    t.insert(
        DETECT_INPUT_FIELD.into(),
        HabValue::IntBuffer(bytemuck::cast_vec(preprocessed_image_buf)),
    );
    t.insert(
        DETECT_SHAPE_FIELD.into(),
        HabValue::ShapeBuffer(preprocessed_image_shape),
    );

    v.push(t);
    v
}

pub const BOXES_BUFFER_FIELD: &str = "bounding_boxes";
pub const BOXES_SHAPE_FIELD: &str = "bbs_shape";
pub const RAW_BOXES_BUFFER_FIELD: &str = "raw_boxes_buffer";
pub const RAW_BOXES_SHAPE_FIELD: &str = "raw_boxes_shape";
pub const NUM_BOXES_FIELD: &str = "num_boxes";

pub const SCORES_BUFFER_FIELD: &str = "scores_buffer";
pub const SCORES_SHAPE_FIELD: &str = "scores_shape";

pub fn postprocess_boxes(mut t: Tuple) -> Vec<Tuple> {
    let mut v = get_tuple_vec();

    let Some(original_image_shape) = t.get(ORIGINAL_IMAGE_SHAPE_FIELD) else {
        error!("Failed to extract original image shape");
        return v;
    };

    let Some(original_image_shape) = original_image_shape.as_shape_buffer() else {
        error!("Failed to extract original image shape");
        return v;
    };

    let width = original_image_shape[0] as u32;
    let height = original_image_shape[1] as u32;
    debug!(
        "starting postprocessing for image with shape {:?}, which should be equal to ({:?}x{:?}x3)",
        original_image_shape,
        t.get("original_width"),
        t.get("original_height"),
    );

    let Some(box_buffer) = t.get(RAW_BOXES_BUFFER_FIELD) else {
        error!("Failed to extract boxes");
        return v;
    };
    let Some(boxes_shape) = t.get(RAW_BOXES_SHAPE_FIELD) else {
        error!("Failed to extract boxes shape");
        return v;
    };
    let Some(box_buffer) = box_buffer.as_int_buffer() else {
        error!("Failed to extract boxes");
        return v;
    };
    let box_buffer = bytemuck::cast_slice(box_buffer);
    let Some(boxes_shape) = boxes_shape.as_shape_buffer() else {
        error!("Failed to extract boxes shape");
        return v;
    };

    let Some(scores_buffer) = t.get(SCORES_BUFFER_FIELD) else {
        error!("Failed to extract scores");
        return v;
    };
    let Some(scores_shape) = t.get(SCORES_SHAPE_FIELD) else {
        error!("Failed to extract scores shape");
        return v;
    };
    let Some(scores_buffer) = scores_buffer.as_int_buffer() else {
        error!("Failed to extract scores");
        return v;
    };
    let scores_buffer = bytemuck::cast_slice(scores_buffer);
    let Some(scores_shape) = scores_shape.as_shape_buffer() else {
        error!("Failed to extract scores shape");
        return v;
    };
    // use the raw boxes that we have right now to compute the real ones
    let (boxes, _labels, _probs) = postprocess(
        scores_buffer,
        scores_shape,
        box_buffer,
        boxes_shape,
        width as i32,
        height as i32,
        PROB_THRESHOLD,
    );
    let boxes_shape = boxes.shape().to_vec();
    let num_boxes = boxes.shape()[0];
    debug!(
        "postprocessing after detection found {} boxes for tuple {} (image id {:?})",
        num_boxes,
        t.id(),
        t.get(crate::face_utils::ORIGINAL_IMAGE_ID_FIELD)
    );
    let (boxes_buf, extra) = boxes.into_raw_vec_and_offset();
    if let Some(extra @ 1..) = extra {
        error!("Failed to extract float boxes buffer: {:?}", extra);
        return v;
    }
    t.insert(BOXES_BUFFER_FIELD.into(), HabValue::IntBuffer(boxes_buf));
    t.insert(BOXES_SHAPE_FIELD.into(), HabValue::ShapeBuffer(boxes_shape));
    t.insert(NUM_BOXES_FIELD.into(), HabValue::Integer(num_boxes as i32));
    t.insert(
        face_utils::FACES_DETECTED_FIELD.into(),
        HabValue::Integer(num_boxes as i32),
    );
    t.insert(
        face_utils::FACE_COUNT_FIELD.into(),
        HabValue::Integer(num_boxes as i32),
    );

    v.push(t);
    v
}

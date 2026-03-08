// TODO:  go back through and implement the efficient ndarray version of this

use anyhow::bail;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};

// A simple language for creating buckets and executing pre-classifiers based on that.
// use std::collections::BTreeMap;
// use crate::HabString;
// use crate::async_query_builder::{FunctionKinds, FunctionLookup};
// use ndarray::{Array1, Array2};
// use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};

// enum PreClassifer {
//     Static(f32),
//     Binary(BinaryLogReg),
//     IndividualBinary(Vec<BinaryLogReg>),
// }

// struct BinaryLogReg {
//     // default 0.5
//     threshold: f32,
//     weights: Array1,
//     output: Array1,
//     intercept: f32,
// }

// fn probabilisitic

use std::sync::Arc;

use crate::scheduler::ShareableArray;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum PreClassifer {
    Static {
        class: usize,
    },
    #[serde(rename = "binary_logistic_regression")]
    Binary {
        #[serde(flatten)]
        inner: BinaryLogReg,
    },
    #[serde(rename = "n_way_binary_logistic_regression")]
    IndividualBinary {
        components: Arc<[BinaryLogReg]>,
    },
    #[serde(rename = "multiclass_logistic_regression")]
    Multiclass {
        #[serde(flatten)]
        inner: MultiLogReg,
    },
}

impl PreClassifer {
    pub fn get_class(&self, inputs: &[f32]) -> usize {
        match self {
            PreClassifer::Static { class } => *class,
            PreClassifer::Binary { inner: classifier } => classifier.get_class(inputs) as usize,
            PreClassifer::IndividualBinary {
                components: classifiers,
            } => {
                let mut class = 0usize;
                for classifier in classifiers.iter() {
                    class = class << 1;
                    let bit = classifier.get_class(inputs);
                    class |= bit as usize;
                }
                class
            }
            PreClassifer::Multiclass { inner: classifier } => classifier.get_class(inputs),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryLogReg {
    // default 0.5
    #[serde(default = "default_threshold")]
    threshold: f32,
    num_inputs: usize,
    weights: Arc<[f32]>,
    intercept: f32,
}
fn default_threshold() -> f32 {
    0.5
}

impl BinaryLogReg {
    pub fn new(weights: Arc<[f32]>, intercept: f32) -> Self {
        Self {
            num_inputs: weights.len(),
            threshold: 0.5,
            weights,
            intercept,
        }
    }

    pub fn with_threshold(weights: Arc<[f32]>, intercept: f32, threshold: f32) -> Self {
        Self {
            num_inputs: weights.len(),
            threshold,
            weights,
            intercept,
        }
    }

    pub fn get_class(&self, inputs: &[f32]) -> bool {
        if inputs.len() != self.num_inputs {
            let msg = format!("different number of inputs found when executing BinaryLogReg. Expected: {}. Found: {}.", self.num_inputs,inputs.len());
            error!("{}", msg);
            panic!("{}", msg);
        }
        self.get_confidence(inputs) > self.threshold
    }

    pub fn get_confidence(&self, inputs: &[f32]) -> f32 {
        if inputs.len() != self.num_inputs {
            let msg = format!("different number of inputs found when executing BinaryLogReg. Expected: {}. Found: {}.", self.num_inputs,inputs.len());
            error!("{}", msg);
            panic!("{}", msg);
        }
        let progress = self.compute(inputs);
        let progress = progress.exp();
        progress / (1.0 + progress)
    }

    pub fn compute(&self, inputs: &[f32]) -> f32 {
        if inputs.len() != self.num_inputs {
            let msg = format!("different number of inputs found when executing BinaryLogReg. Expected: {}. Found: {}.", self.num_inputs,inputs.len());
            error!("{}", msg);
            panic!("{}", msg);
        }
        self.weights
            .iter()
            .zip(inputs)
            .map(|(x, y)| x * y)
            .sum::<f32>()
            + self.intercept
    }

    pub fn make_buckets(&self, inputs: &[f32], num_buckets: usize) -> Vec<(usize, f32)> {
        if inputs.len() != self.num_inputs {
            let msg = format!("different number of inputs found when executing BinaryLogReg. Expected: {}. Found: {}.", self.num_inputs,inputs.len());
            error!("{}", msg);
            panic!("{}", msg);
        }
        let mut buckets = vec![(0, 0.0); num_buckets];
        for i in 0..num_buckets {
            buckets[i].0 = i;
            buckets[i].1 = self.get_confidence(inputs);
        }
        buckets
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLogReg {
    num_inputs: usize,
    weights: Arc<Vec<Vec<f32>>>,
    #[serde(rename = "intercept")]
    intercepts: Arc<[f32]>,
}

impl MultiLogReg {
    pub fn new(weights: Arc<Vec<Vec<f32>>>, intercepts: Arc<[f32]>) -> Self {
        let mut input_counts = weights.iter().map(|w| w.len()).enumerate();
        let Some((_, num_inputs)) = input_counts.next() else {
            let msg = format!("no inputs found when creating MultiLogReg");
            error!("{}", msg);
            panic!("{}", msg);
        };
        if let Some((mismatched_id, mismatched_len)) = input_counts.find(|(_, v)| *v != num_inputs)
        {
            let msg = format!("different number of inputs found when creating MultiLogReg. Expected: {num_inputs}. Found: {mismatched_len} at position {mismatched_id}.");
            error!("{}", msg);
            panic!("{}", msg);
        }
        Self {
            num_inputs,
            weights,
            intercepts,
        }
    }

    pub fn get_class(&self, inputs: &[f32]) -> usize {
        if inputs.len() != self.num_inputs {
            let msg = format!("different number of inputs found when executing MultiLogReg. Expected: {}. Found: {}.", self.num_inputs,inputs.len());
            error!("{}", msg);
            panic!("{}", msg);
        }
        let Some((classno, _conf)) = self
            .get_confidences(inputs)
            // technically there's no need for the confidences since they are monotonic with the raw output so we could do this instead
            // .compute(inputs)
            .enumerate()
            .max_by(|(_idx, confx), (_idy, confy)| confx.total_cmp(confy))
        else {
            error!("encountered empty iterator");
            return 0;
        };
        classno
    }

    pub fn get_confidences<'a>(&'a self, inputs: &'a [f32]) -> impl 'a + Iterator<Item = f32> {
        if inputs.len() != self.num_inputs {
            let msg = format!("different number of inputs found when executing MultiLogReg. Expected: {}. Found: {}.", self.num_inputs,inputs.len());
            error!("{}", msg);
            panic!("{}", msg);
        }
        self.compute(inputs).map(|progress| {
            let progress = progress.exp();
            progress / (1.0 + progress)
        })
    }

    pub fn compute<'a>(&'a self, inputs: &'a [f32]) -> impl 'a + Iterator<Item = f32> {
        if inputs.len() != self.num_inputs {
            let msg = format!("different number of inputs found when executing MultiLogReg. Expected: {}. Found: {}.", self.num_inputs,inputs.len());
            error!("{}", msg);
            panic!("{}", msg);
        }
        self.weights
            .iter()
            .zip(self.intercepts.iter())
            .map(move |(weights, intercept)| {
                weights.iter().zip(inputs).map(|(x, y)| x * y).sum::<f32>() + intercept
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketData {
    pub class: usize,
    pub rewards: Arc<[f64]>,
    pub costs: Arc<[f64]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PreclassifierLangClass(usize);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileFormat {
    pub preclassifier: PreClassifer,
    pub buckets: Arc<[BucketData]>,
}

#[derive(Debug, Clone)]
pub struct RealBucketLookup {
    pub preclassifier: PreClassifer,
    pub buckets: Arc<[crate::scheduler::BinInfo<PreclassifierLangClass>]>,
}

pub fn load_file_format(bytes: &[u8]) -> anyhow::Result<RealBucketLookup> {
    let file_format: FileFormat = serde_json::from_slice(bytes)?;
    // validate lengths
    match &file_format.preclassifier {
        PreClassifer::Static { .. } => {}
        PreClassifer::Binary { inner } => {
            validate_binary_logistic_regression(inner)?;
        }
        PreClassifer::IndividualBinary { components: inner } => {
            let num_inputs = inner[0].num_inputs;
            for classifier in inner.iter() {
                if num_inputs != classifier.num_inputs {
                    let msg = format!("different number of inputs found when validating individual binary classifiers. Expected: {}. Found: {}.", num_inputs, classifier.num_inputs);
                    error!("{}", msg);
                    bail!(msg);
                }
                validate_binary_logistic_regression(classifier)?;
            }
        }
        PreClassifer::Multiclass { inner } => {
            if inner.weights.len() != inner.intercepts.len() {
                let msg = format!("different number of weights and intercepts found when validating multiclass classifier. Expected: {}. Found: {}.", inner.weights.len(), inner.intercepts.len());
                error!("{}", msg);
                bail!(msg);
            }
            let num_inputs = inner.num_inputs;
            for classifier in inner.weights.iter() {
                if num_inputs != classifier.len() {
                    let msg = format!("different number of inputs found when validating multiclass classifier. Expected: {}. Found: {}.", num_inputs, classifier.len());
                    error!("{}", msg);
                    bail!(msg);
                }
            }
        }
    };
    // now that it's validated, we can convert the buckets
    let mut buckets = Vec::with_capacity(file_format.buckets.len());
    let mut max_class = 0;
    for bucket in file_format.buckets.iter() {
        let mut rewards = Vec::new();
        let mut costs = Vec::new();
        // if we don't have a 0 pipeline, we need to add one so we can drop stuff
        let mut valid_pipelines = Vec::new();
        if bucket.costs[0] != 0.0 {
            rewards.push(0.0);
            costs.push(0.0);
        }
        rewards.extend(bucket.rewards.iter().copied());
        costs.extend(bucket.costs.iter().copied());
        valid_pipelines.extend(0..rewards.len());
        max_class = max_class.max(bucket.class);
        let bucket = crate::scheduler::BinInfo {
            id: Some(PreclassifierLangClass(bucket.class)),
            valid_pipelines: ShareableArray::Shared(valid_pipelines.into()),
            rewards: ShareableArray::Shared(rewards.into()),
            costs: ShareableArray::Shared(costs.into()),
        };
        info!("decoded bucket: {:?}", bucket);
        buckets.push(bucket);
    }
    if max_class != buckets.len() - 1 {
        let msg = format!(
            "number of classes does not match number of buckets. Expected: {}. Found: {}.",
            max_class,
            buckets.len()
        );
        error!("{}", msg);
        bail!(msg);
    }

    Ok(RealBucketLookup {
        preclassifier: file_format.preclassifier,
        buckets: Arc::from(buckets),
    })
}

fn validate_binary_logistic_regression(inner: &BinaryLogReg) -> anyhow::Result<()> {
    let num_inputs = inner.num_inputs;
    if num_inputs != inner.weights.len() {
        let msg = format!("inputs and weights do not match when validating BinaryLogReg. Expected: {}. Found: {}.", num_inputs,inner.weights.len());
        error!("{}", msg);
        bail!(msg);
    }

    Ok(())
}

pub fn push_input_to_buckets(
    inputs: &[f32],
    preclassifier: &RealBucketLookup,
    out_buckets: &mut Vec<crate::scheduler::BinInfo<PreclassifierLangClass>>,
) {
    let class = preclassifier.preclassifier.get_class(inputs);
    for bucket in preclassifier.buckets.iter() {
        if bucket.id == Some(PreclassifierLangClass(class)) {
            out_buckets.push(bucket.clone());
            break;
        }
    }
}

pub fn map_inputs_to_bucket(
    inputs: &[f32],
    preclassifier: &RealBucketLookup,
) -> crate::scheduler::BinInfo<PreclassifierLangClass> {
    let class = preclassifier.preclassifier.get_class(inputs);
    preclassifier.buckets[class].clone()
}

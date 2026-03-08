use anyhow::Context;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, PartialEq, PartialOrd, Serialize, Deserialize, Clone, Copy)]
pub enum DistanceMetric {
    #[serde(rename = "euclidean")]
    Euclidean,
    #[serde(rename = "cosine")]
    Cosine,
}

#[derive(Debug, PartialEq, PartialOrd, Serialize, Deserialize, Clone)]
struct IdQueryResponse {
    id: String,
}

pub async fn get_collection_id(
    url: &str,
    collection_name: &str,
    url_string_out: &mut String,
    client_out: &mut Option<reqwest::Client>,
) -> anyhow::Result<String> {
    let client = client_out.get_or_insert_with(|| reqwest::Client::new());
    {
        url_string_out.clear();
        use std::fmt::Write;
        write!(url_string_out, "{url}/api/v1/collections/{collection_name}")?;
    }
    debug!("url to request in get_collection_id: {url_string_out}");
    let response = client
        .get(&*url_string_out)
        .send()
        .await
        .context("failed to send request for collection id")?;
    let IdQueryResponse { id } = response
        .json()
        .await
        .context("failed to parse response for collection id")?;
    Ok(id)
}

#[derive(Debug, Serialize)]
struct Query<'a, List, Str> {
    query_embeddings: &'a [List],
    n_results: usize,
    include: &'a [Str],
}

#[derive(Debug, Deserialize, Serialize)]
pub struct QueryResponse {
    pub ids: Vec<Vec<String>>,
    pub embeddings: Option<Vec<Vec<Vec<f32>>>>,
    pub documents: Option<Vec<Vec<serde_json::Value>>>,
    // uris: Option<Vec<String>>,
    pub metadatas: Option<Vec<Vec<Option<serde_json::Map<String, serde_json::Value>>>>>,
    pub distances: Option<Vec<Vec<f32>>>,
}

pub async fn query_collection<F, L, S>(
    url: &str,
    collection_id: &str,
    embeddings: &[L],
    n_results: usize,
    client_out: &mut Option<reqwest::Client>,
    url_string_out: &mut String,
    include: &[S],
) -> anyhow::Result<QueryResponse>
where
    L: Serialize,
    for<'a> &'a L: IntoIterator<Item = &'a F>,
    F: Serialize + Debug + num_traits::Float,
    S: Serialize + AsRef<str>,
{
    let client = client_out.get_or_insert_with(|| reqwest::Client::new());
    url_string_out.clear();
    {
        use std::fmt::Write;
        write!(
            url_string_out,
            "{url}/api/v1/collections/{collection_id}/query"
        )?;
    }
    let q = Query {
        query_embeddings: embeddings,
        n_results,
        include,
    };
    // println!("sending query to {url_string_out}:\n{:?}\n", serde_json::to_string(&q));
    // info!("sending query to {url_string_out}:\n{:?}\n", serde_json::to_string(&q));
    let response = client
        .post(&*url_string_out)
        .json(&q)
        .send()
        .await
        .context("failed to send request for query")?;
    let response = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("failed to get body from collection query response as bytes: {e}");
            return Err(anyhow::anyhow!(
                "failed to get bytes from collection query response: {e}"
            ));
        }
    };
    let response = match serde_json::from_slice(response.as_ref()) {
        Ok(v) => v,
        Err(e) => {
            error!("failed to parse chroma collection response: {e}");
            match std::str::from_utf8(response.as_ref()) {
                Ok(content) => debug!("content that failed to parse:\n{content:?}\n",),
                Err(e) => {
                    error!("could not parse collection query response as a string at all: {e}")
                }
            }
            return Err(anyhow::anyhow!(
                "failed to parse chroma collection query response: {e}"
            ));
        }
    };
    Ok(response)
}

pub async fn e2e_query<F, L, S>(
    url: &str,
    collection_name: &str,
    embeddings: &[L],
    n_results: usize,
    client_out: &mut Option<reqwest::Client>,
    url_string_out: &mut String,
    include: &[S],
) -> anyhow::Result<QueryResponse>
where
    L: Serialize,
    for<'a> &'a L: IntoIterator<Item = &'a F>,
    F: Serialize + Debug + num_traits::Float,
    S: Serialize + AsRef<str>,
{
    let collection_id = get_collection_id(url, collection_name, url_string_out, client_out).await?;
    query_collection(
        url,
        &collection_id,
        embeddings,
        n_results,
        client_out,
        url_string_out,
        include,
    )
    .await
}

// TODO: custom implementation of upsert
pub async fn upsert(
    _ids: Vec<impl AsRef<str>>,
    _embeddings: Vec<Vec<f32>>,
    _metadata: Option<Vec<serde_json::Map<String, serde_json::Value>>>,
) -> anyhow::Result<()> {
    Err(anyhow::anyhow!(
        "all data should already be inserted into chromadb"
    ))
}

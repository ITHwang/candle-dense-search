use candle::Tensor;
use models_hf::bert::BertInferenceModel;
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: embedding_generator <file_name>");
        std::process::exit(1);
    }

    let file_name = &args[1];
    println!("Starting to generate embeddings from file {}", file_name);

    // get the titles of papers in the csv file(arxiv_data.csv)
    let sentences = get_textcsv_as_vec(file_name, 0).unwrap();
    println!("{} sentences loaded", sentences.len());

    // serialize the sentences to a file
    let mut file = std::fs::File::create("text_map.bin").unwrap();
    bincode::encode_into_std_write(&sentences, &mut file, bincode::config::standard())
        .expect("Failed to encode sentences");
    println!("text_map serialized to text_map.bin");

    // Load the bert model
    let bert_model = BertInferenceModel::load(
        "sentence-transformers/all-MiniLM-L6-v2",
        "refs/pr/21",
        "",
        "",
    )
    .unwrap();
    println!("Bert model loaded");

    // Generate embeddings
    let results: Vec<Result<Tensor, _>> = sentences
        // Batch size: 8
        .par_chunks(8)
        .map(|chunk| bert_model.create_embeddings(chunk.to_vec()))
        .collect();

    let embeddings = Tensor::cat(
        &results
            .iter()
            .map(|r| r.as_ref().unwrap())
            .collect::<Vec<_>>(),
        0,
    )
    .unwrap();

    println!("Embeddings generated");

    // Serialize the embeddings to a file
    embeddings
        .save_safetensors("my_embedding", "embeddings.bin")
        .unwrap();
    println!("embeddings.bin saved");
}

fn get_textcsv_as_vec(filename: &str, col_index: usize) -> anyhow::Result<Vec<String>> {
    // NOTE: only embed the first 3000 titles of arxiv papers
    let result = csv::Reader::from_path(filename)
        .unwrap()
        .records()
        .take(3000)
        .map(|record| record.unwrap().get(col_index).unwrap().to_string())
        .collect();

    Ok(result)
}

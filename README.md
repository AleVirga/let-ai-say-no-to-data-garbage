# Let AI Say No to Data Garbage

## Project Description
In the data and analytics community, high-quality data is widely recognized as the foundation of effective AI models and insightful analytics. Yet, ensuring data quality often requires significant human effort—effort that is time-consuming, costly, and frequently tedious. This project addresses the challenge by demonstrating how AI can intelligently reject low-quality or erroneous data, automating and enhancing data quality control processes.

Leveraging both unsupervised and supervised learning techniques, our proposed AI framework is designed to improve accuracy and reduce manual intervention in heterogeneous and correlated datasets. Using geospatial location data as a representative example, this project tackles the limitations of traditional anomaly detection approaches, which often struggle with datasets containing both spatial and categorical variables with dependencies.

**Key Features**:
- **Integration of Clustering Techniques**: To address spatial and categorical dependencies in the data.
- **String Similarity-Based Standardization**: Improves consistency and reduces errors in text data.
- **Partitioned Isolation Forest**s: Enhances anomaly detection for datasets with complex correlations.
- **Supervised Feedback Loop**: Refines results through iterative learning, reducing false positives and achieving high precision.
- **Scalable Framework**: Designed to support organizations' data quality control and data governance frameworks.
### Why It Matters:
As data-driven transformations become critical to business success, companies are prioritizing intelligent data quality control within their data governance strategies. This solution not only minimizes manual efforts but also ensures that only high-quality data feeds into AI models and analytics pipelines—elevating it to a top-tier data strategy.

## Installation

1. Clone the repository:

```bash 
git clone https://github.com/your-repo-url/let-ai-say-no-to-data-garbage.git
```

2. Navigate to the project directoty:

```bash 
cd let-ai-say-no-to-data-garbage
```

3. Install required dependancies:

```bash 
pip install -r requirements.txt
```

## Usage
The code can be executed through the notebooks located in the `Notebooks` folder. Each notebook is numbered to indicate the proper order of execution.
We recommend preparing your dataset with the following columns:

- `id__cmd`: A unique identifier (integer).
- `lat__cmd`: Latitude (decimal).
- `long__cmd`: Longitude (decimal).
- `city__cmd`: A categorical feature (string).

_Note: The categorical feature (_`city__cmd`_) can be any variable of your choice. You may need to adjust the code in the notebooks to suit your dataset.
Same adjustment must be applied to the name of the input table._


## License
See the [LICENSE](https://github.com/AleVirga/let-ai-say-no-to-data-garbage/blob/main/LICENSE.md) file for details.

## Acknowledgments
Link to the ArXiv paper will be added soon!

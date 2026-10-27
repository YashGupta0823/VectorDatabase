1) AlphaSearch is a research tool that helps users find academic papers related to trading, investing, and finance by searching for ideas rather than exact keywords. I chose this project because I am interested in quantitative finance and wanted to make it easier to find and explore research without knowing the exact financial terms used in academic papers.

2) This stores 600 real academic research papers collected from arXiv across six quantitative finance categories. Each paper is stored as one item, using its title and abstract rather than the full paper. The database also stores information such as the paper's authors, publication date, category, arXiv ID, and URL. Users can also add their own research notes, which are stored in the same database.

3) Users can run the application through the terminal and select options from a menu. To add information, they select "Add Research" and enter a title, description, and optional details. The new information is saved immediately and can be searched even after closing and reopening the program. To search, users can type a question or describe a topic they are interested in. This finds the most relevant papers based on their meaning and can create a research brief comparing the papers and suggesting further research questions.

4)
Query: "Who covers the bill when a hurricane strikes, and can outside investors carry that danger?"
Retrieved paper: Design and pricing of a transparent parametric-modeled loss CAT bond
This search finds research about catastrophe bonds, which allow financial risks from disasters to be transferred to investors. A normal keyword search would struggle because the question uses everyday language, while the paper uses technical financial terms.

Query: "What happens when frightened customers all rush to withdraw their savings at once?"
Retrieved paper: Bank Run Exposure in a Paycheck-to-Paycheck Economy with Loss-Averse Depositors
This question describes a bank run without using the actual term "bank run." AlphaSearch is able to recognize the meaning of the question and retrieve the relevant paper, even though a normal keyword search would miss it because the wording is different.

5)  The research papers come from arXiv, a free online collection of academic papers. 
The application downloads the papers from arXiv and adds them to the local vector database. Running the import again will skip papers that are already stored.

6) This uses ChromaDB as its vector database and the `all-MiniLM-L6-v2` embedding model to convert text into numerical representations called embeddings. These embeddings allow the application to compare the meaning of a user's question with the research papers in the database.
The embedding model downloads automatically the first time it is needed. After the initial setup and dataset import, the database is stored locally and can be searched without an internet connection.

7)
Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Load the research papers:

```bash
python app.py import --ids-file corpus_ids.txt
```

Start the application:

```bash
python app.py
```

The interactive menu lets users search papers, generate research briefs, add research notes, find related papers, and view database statistics.

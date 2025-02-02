.. AutoRAG documentation master file, created by
   sphinx-quickstart on Wed Jan 17 20:55:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. meta::
   :title: AutoRAG docs documentation
   :description: AutoRAG - AutoML tool for RAG official developer documentation docs
   :keywords: AutoRAG,Auto RAG,RAG,AutoRAG docs,LLM,RAG optimization
   :robots: index,nofollow

########################
AutoRAG documentation
########################

RAG AutoML tool for automatically finds an optimal RAG pipeline for your data.


🤷‍♂️ Why AutoRAG?
************************************

There are numerous RAG pipelines and modules out there,
but you don’t know what pipeline is great for “your own data” and "your own use-case."
Making and evaluating all RAG modules is very time-consuming and hard to do.
But without it, you will never know which RAG pipeline is the best for your own use-case.

That's where **AutoRAG** comes in.

🤸‍♂️ How can AutoRAG helps?
************************************

AutoRAG is a tool for finding optimal RAG pipeline for “your data.”
You can evaluate various RAG modules automatically with your own evaluation data,
and find the best RAG pipeline for your own use-case.

AutoRAG supports

- **Data Creation**: Create RAG evaluation data with your own raw documents.
- **Optimization**: Automatically run experiments to find the best RAG pipeline for your own data.
- **Deployment**: Deploy the best RAG pipeline with single yaml file. Supports Flask server as well.

🚀 AutoRAG GUI
************************************

AutoRAG GUI is a web-based tool for running AutoRAG experiments.
If you are not familiar with command-line interface and YAML file configuration,
you can use AutoRAG GUI to run AutoRAG experiments.

Check out `AutoRAG GUI Docs <https://docs.auto-rag.com/gui/gui.html>`__ for more information.

🏃‍♂️ Getting Started
************************************

``pip install AutoRAG``


In our documentation, we will guide you through the process of `installation <https://docs.auto-rag.com/install.html>`__ and `tutorial <https://docs.auto-rag.com/tutorial.html>`__ for AutoRAG starter.
After you find your first RAG pipeline with AutoRAG, you can learn how to read result files at `here <https://docs.auto-rag.com/structure.html>`__.

And do you want to get the ultimate performance RAG pipeline?
Learn how make great evaluation dataset with your own raw documents at `here <https://docs.auto-rag.com/data_creation/tutorial.html>`__.

Also, you can learn how to set various experiment configurations at `optimization <https://docs.auto-rag.com/optimization/optimization.html>`__ guide.

Of course, you can use your own local LLM or embedding model with AutoRAG. Go to `here <https://docs.auto-rag.com/local_model.html>`__ to learn how to use your own model with AutoRAG.

If you face any trouble? Check out our `troubleshooting <https://docs.auto-rag.com/troubleshooting.html>`__ guide.
Also, feel free to ask your question at our `github issue <https://github.com/Marker-Inc-Korea/AutoRAG/issues>`__ or `Discord <https://discord.gg/P4DYXfmSAs>`__ channel.

🗣️ Talk with Founders
************************************
* Jeffrey Kim : https://zcal.co/autorag-jeffrey/autorag-demo-15min
* Bobb Kim : https://zcal.co/i/tcuLtmq5

👨‍👩‍👧‍👦 Ecosystem
************************************
* Github Repo : https://github.com/Marker-Inc-Korea/AutoRAG
* HomePage : https://auto-rag.com
* PyPI : https://pypi.org/project/AutoRAG/
* Discord : https://discord.gg/P4DYXfmSAs
* Roadmap : https://github.com/orgs/Auto-RAG/projects/1/views/2
* HuggingFace : https://huggingface.co/AutoRAG
* LinkedIn : https://www.linkedin.com/company/autorag/
* X (Twitter) : https://twitter.com/AutoRAG_HQ


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   install.md
   tutorial.md
   structure.md
   troubleshooting.md
   local_model.md
   migration.md
   test_your_rag.md

.. toctree::
    :maxdepth: 1
    :caption: AutoRAG GUI
    :hidden:

    gui/gui.md

.. toctree::
   :maxdepth: 1
   :caption: Vector DB
    :hidden:

    vectordb/vectordb.md

.. toctree::
   :maxdepth: 2
   :caption: Data Creation
   :hidden:

   data_creation/tutorial.md
   data_creation/data_format.md
   data_creation/data_creation.md


.. toctree::
   :maxdepth: 2
   :caption: Optimization
   :hidden:

   optimization/optimization.md
   optimization/folder_structure.md
   optimization/custom_config.md
   optimization/strategies.md
   optimization/sample_config.md

.. toctree::
   :maxdepth: 3
   :caption: Integration
   :hidden:

   integration/llm/llm.md
   integration/vectordb/vectordb.md


.. toctree::
   :maxdepth: 2
   :caption: Evaluation Metrics
   :hidden:

   evaluate_metrics/retrieval.md
   evaluate_metrics/retrieval_contents.md
   evaluate_metrics/generation.md


.. toctree::
   :maxdepth: 3
   :caption: Nodes & Modules
   :hidden:

   nodes/index.md
   nodes/query_expansion/query_expansion.md
   nodes/retrieval/retrieval.md
   nodes/passage_augmenter/passage_augmenter.md
   nodes/passage_reranker/passage_reranker.md
   nodes/passage_filter/passage_filter.md
   nodes/passage_compressor/passage_compressor.md
   nodes/prompt_maker/prompt_maker.md
   nodes/generator/generator.md

.. toctree::
   :maxdepth: 2
   :caption: Deploy
   :hidden:

   deploy/api_endpoint.md
   deploy/web.md

.. toctree::
   :maxdepth: 1
   :caption: Roadmap
   :hidden:

   roadmap/modular_rag.md

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_spec/modules

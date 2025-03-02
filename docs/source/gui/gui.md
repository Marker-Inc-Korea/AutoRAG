---
myst:
   html_meta:
      title: AutoRAG GUI
      description: Learn how to use AutoRAG GUI
      keywords: AutoRAG,RAG,AutoRAG GUI,RAG GUI
---
# AutoRAG GUI

AutoRAG GUI is a web-based GUI for AutoRAG.
If AutoRAG is a little bit complicated to you, try AutoRAG GUI.

Your Optimized RAG pipeline is just a few clicks away.

|                                    Project Management                                     |                                    Easy Configuration                                     |                                     Parsed Page View                                      |
|:-----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|
| ![Image](https://github.com/user-attachments/assets/87289d84-ff65-4810-bc41-3f30b36b7ddf) | ![Image](https://github.com/user-attachments/assets/dbe0a49b-ebf2-4c9c-b17d-1be1c2cd1060) | ![Image](https://github.com/user-attachments/assets/d8a50512-3299-4b68-b48e-e2f49d688f01) |

## Installation

### Use Docker Compose (from source)
You can install AutoRAG GUI using Docker and the source from Github.

#### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/Marker-Inc-Korea/AutoRAG.git
cd AutoRAG
```

#### 2. Run Docker Compose

```bash
docker compose up
```

#### 3. Access the GUI

You can use the GUI by accessing `http://localhost:3000` in your web browser right away.

The project files will be stored in the 'projects' directory in the root directory that you cloned the repository.

## Newbie & Pro version

There are two versions of AutoRAG GUI: Newbie and Pro.
The Pro version is under construction.

The Newbie version is for the users who are not familiar with RAG or AI at all.
It provides a simple setting to optimize the RAG pipeline and use it.
If you are a newbie, feel free to check out the Newbie version.

## Run the GUI application from source

#### 1. Build an GUI application and run it

First, go to the autorag-frontend directory.

```bash
cd autorag-frontend
```

Then, configure the `.env.local` file.
The example of it is in the `.env.local.example` file.

If you serve the AutoRAG API server to the other address which is not localhost,
you should change the `NEXT_PUBLIC_HOST_URL` and `NEXT_PUBLIC_API_URL` in the `.env.local` file.

After that, run the following commands for building and running the GUI application.

```bash
yarn install
npm run build
npm run start
```

That's it!

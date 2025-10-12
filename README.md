# FastAPI Application Setup

This repository contains a FastAPI application. Follow the steps below to set up and run the application locally.

## Prerequisites

- Python 3.10 or higher  
- Poetry

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tushargoplani/embedding-model-evaluator.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd your-repo
    ```

3. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
    ```

4. **Install the project dependencies using Poetry:**
    ```bash
    poetry install
    ```

5. **Create a `.env` file in the root of the project with the following variables:**
    ```env
    PORT=8000
    COHERE_API_KEY=your_cohere_api_key_here
    VOYAGE_API_KEY=your_voyage_api_key_here
    ```
    > Replace `your_cohere_api_key_here` and `your_voyage_api_key_here` with your actual API keys.

6. **Run the application:**
    ```bash
    python -m app.main
    ```

## Usage

Once the application is running, you can access it at:  
[http://localhost:8000](http://localhost:8000)

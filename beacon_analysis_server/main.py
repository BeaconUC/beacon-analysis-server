from fastapi import FastAPI

from beacon_analysis_server.config import logger

app = FastAPI(
    title="Beacon Analysis Server",
    version="0.0.1",
    description="A FastAPI server serving Beacon analysis logic",
)


@app.get("/")
async def root():
    return {"message": "Beacon Analysis Server is online"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "python_version": "3.14"}


# Example placeholder for your analysis
@app.post("/analyze")
async def run_analysis(data: dict):
    logger.info("Analysis requested")
    # This is where you will eventually call your modeling.predict logic
    return {"result": "Analysis logic goes here"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="slideGen API",
    description="API for generating slides from text using AI.",
    version="0.1.0",
)

# CORS Middleware - Cross-Origin Resource Sharing
# Allows requests from browsers running on different domains (like localhost:3000 -> localhost:8000)
# In production, you'd restrict 'origins' to specific domains
origins = ["*"]  # "*" means allow all origins (good for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Allow cookies/auth headers
    allow_methods=["*"],     # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],     # Allow any headers
)

# Root endpoint - Simple welcome message
@app.get("/")
async def root():
    return {"message": "SlideGen API", "status": "running"}


# health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}



from memory_server import app
from config import HOST, PORT
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)


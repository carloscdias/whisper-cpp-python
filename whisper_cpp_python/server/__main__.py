"""Example FastAPI server for whisper.cpp

To run this example:

Then run:
```
uvicorn whisper_cpp_python.server.app:app --reload
```

or

```
python3 -m whisper_cpp_python.server
```

Then visit http://localhost:8000/docs to see the interactive API docs.

"""
import os
import uvicorn

from whisper_cpp_python.server.app import create_app

if __name__ == "__main__":
    app = create_app()

    uvicorn.run(
        app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", 8001))
    )

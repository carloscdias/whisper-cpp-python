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
import argparse
from typing import get_type_hints

from whisper_cpp_python.server.app import create_app, Settings

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    for name, field in Settings.model_fields.items():
        
        # Get field type 
        field_type = get_type_hints(Settings)[name]  
        
        description = field.description
        if field.default is not None and description is not None:
            description += f" (default: {field.default})"

        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field_type, # Use field type here
            help=description
        )

    args = parser.parse_args()
    
    settings = Settings(**{k: v for k, v in vars(args).items() if v is not None})

    app = create_app(settings=settings)

    uvicorn.run(
        app, 
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 8001))
    )
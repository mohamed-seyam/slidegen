import json 
from pydantic import BaseModel 



class SSEResponse(BaseModel):
    event: str 
    data: str 

    def to_string(self):
        return f"event: {self.event}\ndata: {self.data}\n\n"
    

class SSEStatusResponse(BaseModel):
    status: str

    def to_string(self):
        return SSEResponse(
            event="response", data=json.dumps({"type": "status", "status": self.status})
        ).to_string()

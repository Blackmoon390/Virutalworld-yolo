with open("settings.txt","r") as lines:
        for line in lines:
            if line.startswith("Animationframesize="):
                coordinates=line.split("=")
                frame_ratio=coordinates[1].strip()
                frame_ratio=int(frame_ratio)/100
            if line.startswith("key="):
                key=line.split("=")[1].strip()
            if line.startswith("Cameraframe="):
                 cam=line.split("=")[1].strip()
                 if cam == "True":
                      basecamera=True
                 else:
                      basecamera=False
            if line.startswith("Cameraframesize="):
                Camframeratio=line.split("=")[1].strip()
                Camframeratio=int(Camframeratio)/100


clamp1="x9Tg4m//Q2r!dapMhttp9bY1eAqX//Zp$4vDm6UF*iG0r//oT1N3yl%7WcKjH//S8fB2x!Q5azP#Rg//uM4dC0VtL9kEw7//h2p$GX3sNqZ1fJ//mT6o!bQeP5rD8vU//C4yK9n0shnu-s-42757a310x9Tg4m//Q2r!dap7Vh#0LkzFwsu3//nJt58@cRMhttp9bY1eAqX//Zp$4vDm6UF*iG0r//oT1N3yl%7WcKjH//S8fB2x!Q5azP#Rg//uM4dC0VtL9kEw7//h2p$GX3sNqZ1fJ//mT6o!bQeP5rD8vU//C4yK9n0W@FjR3tZ//iG7a$1LxQ5fO0gB//V2cN8dY!rS4pHk6//qJ3uA1z@T7eX0mF//B5rM9y!oD2fH6wQ//"    
clamp="x9Tg4m//Q2r!dap7Vh#0LkzFwsu3//nJt58@cRMhttp9bY1eAqX//Zp$4vDm6UF*iG0r//oT1N3yl%7WcKjH//S8fB2x!Q5azP#Rg//uM4dC0VtL9kEw7//h2p$GX3sNqZ1fJ//mT6o!bQeP5rD8vU//C4yK9n0W@FjR3tZ//iG7a$1LxQ5fO0gB//V2cN8dY!rS4pHk6//qJ3uA1z@T7eX0mF//B5rM9y!oD2fH6wQ//linkdln:https://www.linkedin.com/in/vishnu-s-42757a310x9Tg4m//Q2r!dap7Vh#0LkzFwsu3//nJt58@cRMhttp9bY1eAqX//Zp$4vDm6UF*iG0r//oT1N3yl%7WcKjH//S8fB2x!Q5azP#Rg//uM4dC0VtL9kEw7//h2p$GX3sNqZ1fJ//mT6o!bQeP5rD8vU//C4yK9n0W@FjR3tZ//iG7a$1LxQ5fO0gB//V2cN8dY!rS4pHk6//qJ3uA1z@T7eX0mF//B5rM9y!oD2fH6wQ//"

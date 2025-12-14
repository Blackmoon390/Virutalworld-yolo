with open("settings.txt","r") as lines:
        for line in lines:
            if line.startswith("Animationframesize="):
                coordinates=line.split("=")
                frame_ratio=coordinates[1].strip()
                frame_ratio=int(frame_ratio)/100
            if line.startswith("key="):
                key=line.split("=")[1].strip()
print(key)
# Foodlg-Flask-Docker
Foodlg Flask backend app to host object detection and classification models, as temporary replacement for Rafiki

### Build application (make sure Git LFS is installed)
#### Refer to https://stackoverflow.com/questions/48734119/git-lfs-is-not-a-git-command-unclear/48734334 for installation of Git LFS
```
$ git lfs install
$ git clone https://github.com/vaanforz/Foodlg-Rafiki-Replacement.git
$ docker build -t foodlg_flask .
```

### Run the container
Create a container from the image.
```
$ docker run --name foodlg-flask-container -d -p 5000:5000 foodlg_flask
```

Now visit http://localhost:5000
```
 Flask app has started successfully! 
```

### Experiment with the API (Optional)

Use the `model/predict` endpoint to load a test image and get predicted labels for the image from the API.
The coordinates of the bounding box are returned in the `detection_box` field, and contain the array of normalized
coordinates (ranging from 0 to 1) in the form `[ymin, xmin, ymax, xmax]`.

You can also test it on the command line, for example:

```
$ curl -F "image=@assets/dinner.jpg" -XPOST http://localhost:5000/model/predict
```

You should see a JSON response like that below:

```json
{
  "status": "ok",
  "predictions": [
      {
          "label_id": "1",
          "label": "banana",
          "probability": 0.944034993648529,
          "detection_box": [
              0.1242099404335022,
              0.12507188320159912,
              0.8423267006874084,
              0.5974075794219971
          ]
      },
      {
          "label_id": "18",
          "label": "duck rice",
          "probability": 0.8645511865615845,
          "detection_box": [
              0.10447660088539124,
              0.17799153923988342,
              0.8422801494598389,
              0.732001781463623
          ]
      }
  ]
}
```

# Gradio Demos

Note that all the Gradio demos do some basic parameter sanitization. That means all paths will be resolved and blocked if they point to parent directories (of the current working directory). That mean the demos need their resources not symlinked but copied to work, or change to the base of this repo to execute the python scripts from there!

## Single tool demo

```bash
python examples/gradio/segmentizer.py
```

```bash
python examples/gradio/tokenizer.py
```

```bash
python examples/gradio/lani.py
```

## All LCC tools demo

```bash
python examples/gradio/lcc_demo.py
```

## Development (Hot-Reloading)

Specify the demo block name and the file.

```bash
gradio --demo-name lcc_demo examples/gradio/lcc_demo.py
```

## Docker Deployment

Run the following commands from the root of this repo!

```bash
docker build -f examples/gradio/Dockerfile -t lcc-gradio-demo .
```

```bash
docker run --rm -it -p "8080:7860" --name lcc-gradio-demo lcc-gradio-demo
```

You can map custom resources into the container at `/workspace/resources/`.

Visit http://localhost:8080 to access the Gradio demo.

_Dev hint: use https://github.com/pwaller/docker-show-context to check your `.dockerignore` file._

import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cyy_torch_text",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/torch_text",
    packages=[
        "cyy_torch_text",
        "cyy_torch_text/dataset",
        "cyy_torch_text/model_evaluator",
        "cyy_torch_text/model",
        "cyy_torch_text/tokenizer",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)

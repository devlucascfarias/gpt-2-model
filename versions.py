import pkg_resources

libraries = [
    "Flask",
    "Transformers",
    "Torch"
]

for library in libraries:
    version = pkg_resources.get_distribution(library).version
    print(f"{library}: {version}")

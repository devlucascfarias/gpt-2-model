import pkg_resources

libraries = [
    "Flask",
    "Transformers",
]

for library in libraries:
    version = pkg_resources.get_distribution(library).version
    print(f"{library}: {version}")

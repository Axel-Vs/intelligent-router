import os
import pkg_resources

def get_package_dependencies():
    package_dependencies = {}
    for package in pkg_resources.working_set:
        dependencies = package.requires()
        package_dependencies[package.key] = [dep.key for dep in dependencies]
    return package_dependencies

def generate_condensed_requirements():
    package_dependencies = get_package_dependencies()

    # Remove packages that are dependencies of other packages
    top_level_packages = set(package_dependencies.keys())
    for dependencies in package_dependencies.values():
        top_level_packages -= set(dependencies)

    condensed_requirements = []
    for package in top_level_packages:
        version = pkg_resources.get_distribution(package).version
        condensed_requirements.append(f"{package}=={version}")

    return condensed_requirements

def write_requirements_file(filename, requirements):
    with open(filename, "w") as file:
        for requirement in requirements:
            file.write(requirement + "\n")


if __name__ == "__main__":
    condensed_requirements = generate_condensed_requirements()

    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    requirements_file_path = os.path.join(parent_directory, "requirements.txt")

    write_requirements_file(requirements_file_path, condensed_requirements)
    print(f"requirements.txt file written to {requirements_file_path}.")
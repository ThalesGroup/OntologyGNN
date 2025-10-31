"""OntologyGNN package."""

__all__ = ["__version__", "gnn_model", "ontology_dataloader", "train", "utils"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ontologygnn")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from . import gnn_model, ontology_dataloader, train, utils

from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Union
import os
from datetime import datetime
from matplotlib import pyplot
from torch import Tensor
from torch.nn import Module
from torchview import draw_graph
from overrides import EnforceOverrides

class LMBuilderLogger(ABC):
    """Abstract base class for building custom loggers."""

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> Optional[Union[int, str]]:
        """Return the experiment version."""

    @property
    @abstractmethod
    def root_dir(self) -> Optional[str]:
        """Return the root directory where all versions of an experiment get saved, or `None` if the logger does not
        save data locally."""
        return None

    @property
    @abstractmethod
    def log_dir(self) -> Optional[str]:
        """
        Return the directory where the logger is saved.

        Returns:
            Optional[str]: Directory where the logger is saved, or None if the logger does not save data locally.
        """
        return None

    @abstractmethod
    def log_msg(self, message: str, with_time: bool = True) -> None:
        """
        Log systematic messages, info, warnings.

        Args:
            message (str): Text to log.
            with_time (bool): Whether to log with time or not.
        """
        

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Records metrics. This method logs metrics as soon as it receives them.

        Args:
            metrics (Dict[str, float]): Dictionary with metric names as keys and measured quantities as values.
            step (Optional[int]): Step number at which the metrics should be recorded.
        """


    def log_plot(self, plot: pyplot.plot, plot_name_prefix: str = "plot") -> None:
        """
        Log Matplotlib plot.

        Args:
            plot: Prebuilt Matplotlib.pyplot plot.
            plot_name_prefix: Filename for the plot to be saved.
        """
        plot.savefig(os.path.join(self.log_dir, f"{plot_name_prefix}.png"))


    @abstractmethod
    def log_hparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        """
        Record hyperparameters.

        Args:
            params (Union[Dict[str, Any], Namespace]): Namespace or Dict containing the hyperparameters.
            args: Optional positional arguments, depends on the specific logger being used.
            kwargs: Optional keyword arguments, depends on the specific logger being used.
        """
        pass
        
    
    def log_model_graph(
        self,
        model: Module,
        input_array: Optional[Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
        graph_prefix: str = "model_graph",
        format: str = "png",
        **kwargs
    ) -> None:
        """
        Record a model's computational graph using the Graphviz library.

        Args:
            model (Module): Your custom LLM (subclass of nn.Module).
            input_array (Optional[Tensor]): Input passed to `model.forward`. Defaults to None.
            input_size (Optional[Tuple[int, int]]): Tuple of size of input which model expects.
            graph_prefix (str): Filename of the image to be saved.
            format (str): Format of the image to be saved. Checkout the supported formats at https://graphviz.org/docs/outputs/.
            **kwargs: Additional keyword arguments to customize the graph. Check the documentation at https://github.com/mert-kurttutan/torchview#documentation.
        """
        if input_array is None and input_size is None:
            raise ValueError("Both input_array and input_size cannot be None to visualize the model's computation")
        
        save_path = os.path.join(self.log_dir, f"{graph_prefix}.{format}")

        graph = draw_graph(model, input_data=input_array, input_size=input_size, save_graph=False, **kwargs).visual_graph
        graph_svg = graph.pipe(format=format).decode('utf-8')  # convert to binary data
        with open(save_path, 'wb') as f:
            f.write(graph_svg)

    def get_curr_date_time(self):
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%I:%M:%S %p")
        return date, time
    
    def _handle_value(self, value: Union[Tensor, Any]) -> Any:
        if isinstance(value, Tensor):
            return value.item()
        return value

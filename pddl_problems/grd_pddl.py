from array import array
from functools import partial
from typing import Iterable, Iterator, List, Optional, Set, Tuple
import time
import os

from pddlstream.language.generator import from_fn, from_gen_fn
from pddlstream.language.constants import PDDLProblem
from pddlstream.utils import INF

# solvers
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.incremental import solve_incremental

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from dotenv import load_dotenv
from utils.harm import is_harmful_text, load_harm_detector
load_dotenv()

GRD_GENERATION_LOG_PATH = "outputs/logs/grd/grd_generation_log.txt"


def append_grd_generation_log(message: str) -> None:
    log_dir = os.path.dirname(GRD_GENERATION_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(GRD_GENERATION_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

@torch.no_grad()
def nucleus_tokens(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prefix: str,
    p: float,
    temperature: float,
    device: str,
    top_k: int = 1,
) -> Iterable[int]:
    """
    Returns token ids: top_k highest-prob + top_k lowest-prob within the nucleus set (top-p).
    Minimizes GPU VRAM by moving logits to CPU immediately and doing sorting/softmax on CPU.
    """
    # Tokenize on CPU, then move only input_ids to GPU
    enc = tokenizer(prefix, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # No gradients = less memory
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
        # Move last-token logits to CPU ASAP to free VRAM pressure
        logits = (out.logits[0, -1].float().div(temperature)).cpu()
        del out

    # Heavy operations on CPU
    # Sort logits (not probs) then compute cumulative probs from log-softmax
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Stable + avoids building full softmax on GPU; still CPU here.
    sorted_logprobs = torch.log_softmax(sorted_logits, dim=-1)
    sorted_probs = sorted_logprobs.exp()
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = int(torch.searchsorted(cum_probs, p).item())
    top_p = sorted_indices[: cutoff + 1]

    if top_p.numel() > 2 * top_k:
        sampled = torch.cat([top_p[:top_k], top_p[-top_k:]])
    else:
        sampled = top_p

    # Yield python ints
    for idx in sampled.tolist():
        yield idx

class GRDPDDLStream:
    """PDDLStream formulation for the guard‑rail problem."""

    def _init_node_storage(self) -> None:
        """Initialize the shared token node storage with compact arrays."""
        # Use sentinel values to avoid Python object overhead for None.
        self.NULL_TOKEN = -1
        self.NULL_PARENT = -1
        self.tokens = array("i")
        self.parents = array("i")

    def _add_node(self, token: Optional[int], parent: Optional[int]) -> int:
        """Append a node to the token tree and return its id."""
        node_id = len(self.tokens)
        self.tokens.append(self.NULL_TOKEN if token is None else int(token))
        self.parents.append(self.NULL_PARENT if parent is None else int(parent))
        return node_id

    def _tokens_from_node(self, node_id: int) -> List[int]:
        """Reconstruct tokens by traversing from node to root."""
        tokens: List[int] = []
        cursor: int = node_id
        while cursor != self.NULL_PARENT:
            tok = self.tokens[cursor]
            if tok != self.NULL_TOKEN:
                tokens.append(tok)
            cursor = self.parents[cursor]
        tokens.reverse()
        return tokens

    def _parse_state(self, state: str) -> Tuple[int, int]:
        """Extract the g0/g1 node ids from the encoded state string."""
        g0_part, g1_part = state.split("<GEN_SEP>")
        g0_id = int(g0_part.split("_", 1)[1]) if "_" in g0_part else 0
        g1_id = int(g1_part.split("_", 1)[1]) if "_" in g1_part else 0
        return g0_id, g1_id

    def _build_state(self, g0_id: int, g1_id: int) -> str:
        """Encode node ids into the compact state string used by PDDL objects."""
        return f"g0_{g0_id}<GEN_SEP>g1_{g1_id}"

    def _get_agent_node_id(self, state: str, agent: str) -> int:
        """Return the node id relevant for the requested agent."""
        g0_id, g1_id = self._parse_state(state)
        return g1_id if agent == "1" else g0_id

    def tokens_for_state(self, state: str, agent: str) -> List[int]:
        """Public helper to fetch the token list for a given agent/state."""
        return self._tokens_from_node(self._get_agent_node_id(state, agent))

    def _decode_from_node(self, node_id: int) -> str:
        tokens = self._tokens_from_node(node_id)
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def __init__(
        self,
        start_text: str,
        model,
        tokenizer,
        device,
        nucleus_p: float = 0.9,
        temperature: float = 0.55,
        hf_token: Optional[str] = None,
        harm_detector_path: str = "harm_detector/models/harm_detector.dill",
        max_length: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_frontier_per_depth: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
        use_focused_first: bool = True,
        focused_time_share: float = 0.4,
    ) -> None:
        """
        Initialize a new guard-rail PDDLStream planner.

        Args:
            start_text: The initial prompt text to seed generation.
            model_name: Name of the HuggingFace model to load.
            nucleus_p: Top-p nucleus sampling threshold.
            temperature: Sampling temperature for the LLM.
            hf_token: Optional HuggingFace access token.
            harm_detector_path: Path to a pickled harm detector model.
            max_length: Optional maximum generation length in tokens (L). When set,
                generations that either include an EOS token or reach this length
                are considered terminal states.
            max_nodes: Optional hard cap on total nodes. Leave ``None`` to disable.
            max_frontier_per_depth: Optional per-depth beam limit. Helps keep the
                frontier size tame without outright stopping the run.
            max_memory_mb: Soft memory ceiling (MB) passed to PDDLStream. When set,
                the solver terminates before exceeding this process memory.
            use_focused_first: Try the focused solver before falling back to incremental.
            focused_time_share: Fraction of max_time reserved for focused before falling back.
        """
        self.start_text = start_text
        self.nucleus_p = nucleus_p
        self.temperature = temperature
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.max_memory_kb = (max_memory_mb * 1024) if max_memory_mb is not None else INF
        self.max_frontier_per_depth = max_frontier_per_depth
        self.frontier_counts: dict[int, int] = {}
        self.use_focused_first = use_focused_first
        self.focused_time_share = max(0.0, min(focused_time_share, 1.0))

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._node_limit_reached = False

        self.harm_detector = load_harm_detector(harm_detector_path)

        self._init_node_storage()

        # Encode the starting prompt into a sequence of token IDs without special tokens
        try:
            start_tokens: List[int] = self.tokenizer.encode(start_text, add_special_tokens=False)
        except Exception:
            start_tokens = []

        # Update the maximum length to account for the starting prompt
        self.input_length = len(start_tokens)
        if self.max_length is not None:
            self.max_length += len(start_tokens)

        # Build a compact node-based representation of the starting sequence.
        self.root_node_id = self._add_node(token=None, parent=None)
        current_node = self.root_node_id
        for tok in start_tokens:
            current_node = self._add_node(token=tok, parent=current_node)
        self.start_node_id = current_node

        # Construct the initial PDDL object name using node ids instead of full token strings.
        self.start_obj = self._build_state(self.start_node_id, self.start_node_id)
        self.generated_states: Set[str] = {self.start_obj}


        # Get the directory containing this Python file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(f'{self.base_dir}/pddl_files/grd_domain.pddl'):
            raise FileNotFoundError("Domain PDDL file not found.")
        with open(f'{self.base_dir}/pddl_files/grd_domain.pddl', 'r') as f:
            self.DOMAIN_PDDL = f.read()

        if not os.path.exists(f'{self.base_dir}/pddl_files/grd_streams.pddl'):
            raise FileNotFoundError("Streams PDDL file not found.")
        with open(f'{self.base_dir}/pddl_files/grd_streams.pddl', 'r') as f:
            self.STREAM_PDDL = f.read()

    def next_stream(self, p: str, agent: str) -> Iterator[Tuple[str]]:
        """
        Generate successor prefixes for a given PDDL object name.
        """

        g0_id, g1_id = self._parse_state(p)
        base_node_id = g1_id if agent == "1" else g0_id
        tokens = self._tokens_from_node(base_node_id)

        # already terminated (contain an EOS token) or reached the length limit.
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and eos_id in tokens:
            return

        if self.max_length is not None and len(tokens) >= self.max_length:
            return

        prefix_text = self._decode_from_node(base_node_id)

        for tok_id in nucleus_tokens(
            tokenizer=self.tokenizer,
            model=self.model,
            prefix=prefix_text,
            p=self.nucleus_p,
            temperature=self.temperature,
            device=self.device,
        ):
            candidate_node_id = len(self.tokens)  # tentative id before adding
            if self.max_nodes is not None and candidate_node_id >= self.max_nodes:
                if not self._node_limit_reached:
                    self._node_limit_reached = True
                    print(f"Node budget of {self.max_nodes} reached; halting further expansions.")
                return

            # Depth counting excludes the fixed prompt length
            new_depth = (len(tokens) - self.input_length) + 1
            if (
                self.max_frontier_per_depth is not None
                and self.frontier_counts.get(new_depth, 0) >= self.max_frontier_per_depth
            ):
                continue

            if agent == "1":
                new_slug = self._build_state(g0_id, candidate_node_id)
            elif agent == "0":
                new_slug = self._build_state(candidate_node_id, g1_id)
            elif agent == "01":
                new_slug = self._build_state(candidate_node_id, candidate_node_id)
            else:
                continue

            if new_slug in self.generated_states:
                continue

            new_node_id = self._add_node(tok_id, base_node_id)

            generated_tokens = self._tokens_from_node(new_node_id)[self.input_length:]
            gen_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            self.generated_states.add(new_slug)
            self.frontier_counts[new_depth] = self.frontier_counts.get(new_depth, 0) + 1
            append_grd_generation_log(f"Prefix: {prefix_text} -> Generated: {gen_text}")
            yield (new_slug,)

    def harmful_stream(self, g: str) -> Optional[Tuple[()]]:
        """A PDDLStream test stream that certifies the ``(harmful ?g)`` fact.

        The stream takes a single generation object and returns an empty tuple
        if the associated text is harmful.  Returning ``None`` indicates that
        the fact cannot be certified (i.e., the generation is safe).  Test
        streams must either return a tuple of no outputs or ``None``.
        """
        g0_id, _ = self._parse_state(g)
        tokens = self._tokens_from_node(g0_id)[self.input_length:]  # only consider generated part
        text = self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        if is_harmful_text(self.harm_detector, text, threshold=0.75):
            print('Harmful generation detected for text:', text)
            append_grd_generation_log(f"Harmful generation detected: {text}")
            return tuple()
        return None

    def ended_stream(self, g: str) -> Optional[Tuple[()]]:
        """A PDDLStream test stream that certifies the ``(ended ?g)`` fact.

        The stream takes a single generation object and returns an empty tuple
        if the associated text is ended.  Returning ``None`` indicates that
        the fact cannot be certified (i.e., the generation is not ended).  Test
        streams must either return a tuple of no outputs or ``None``.
        """
        _, g1_id = self._parse_state(g)
        tokens = self._tokens_from_node(g1_id)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        length_limit_reached = self.max_length is not None and len(tokens) >= self.max_length
        if ((eos_id is not None) and (eos_id in tokens)) or length_limit_reached:
            print('Generation ended for text:', self.tokenizer.decode(
                tokens[self.input_length:], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            append_grd_generation_log(
                f"Generation ended: {self.tokenizer.decode(tokens[self.input_length:], skip_special_tokens=False, clean_up_tokenization_spaces=True)}"
            )
            return tuple()
        return None

    def solve(self, max_time: int = 60):
        """Solve the guard‑rail planning problem using PDDLStream.

        Args:
            max_time: Maximum time (in seconds) to spend planning.

        Returns:
            The plan as a list of actions if found, otherwise ``None``.
        """
        # Map stream names in the PDDL to Python generators
        stream_map = {
            "next0": from_gen_fn(partial(self.next_stream, agent='0')),
            "next1": from_gen_fn(partial(self.next_stream, agent='1')),
            "next01": from_gen_fn(partial(self.next_stream, agent='01')),
            "harmful0": from_fn(self.harmful_stream),
            "ended1": from_fn(self.ended_stream),
        }
        init_atoms = [('llm-at0', self.start_obj), ('llm-at1', self.start_obj), ('gen', self.start_obj), ('=', ('total-cost',), 0)]
        goal_atoms = ('and', ('found-harmful0',), ('found-no-harm1',))

        pddl_problem = PDDLProblem(domain_pddl=self.DOMAIN_PDDL,
                                   constant_map={},
                                   stream_map=stream_map,
                                   stream_pddl=self.STREAM_PDDL,
                                   init=init_atoms,
                                   goal=goal_atoms)

        if self.use_focused_first:
            start_ts = time.time()
            focused_budget = max_time * self.focused_time_share
            focused_solution = solve_focused(
                pddl_problem,
                max_time=focused_budget,
                max_memory=self.max_memory_kb,
                unit_costs=True,
                verbose=False,
            )
            if focused_solution.plan is not None:
                return focused_solution
            elapsed = time.time() - start_ts
            max_time = max(0, max_time - elapsed)

        return solve_incremental(
            pddl_problem,
            max_time=max_time,
            max_memory=self.max_memory_kb,
            unit_costs=True,
            verbose=False,
        )

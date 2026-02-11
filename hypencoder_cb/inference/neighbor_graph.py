from typing import Iterable, List, Tuple, Union

import fire
import torch
from tqdm import tqdm
import time

from hypencoder_cb.inference.shared import load_encoded_items_from_disk
from hypencoder_cb.utils.iterator_utils import batchify_slicing
from hypencoder_cb.utils.jsonl_utils import JsonlWriter
from hypencoder_cb.utils.torch_utils import dtype_lookup


def get_embeddings(path: str) -> Tuple[torch.Tensor, List[str], List[str]]:
    encoded_items = load_encoded_items_from_disk(path)

    item_embeddings = torch.stack(
        [torch.tensor(x.representation) for x in encoded_items]
    )
    item_ids = [x.id for x in encoded_items]
    item_texts = [x.text for x in encoded_items]

    return item_embeddings, item_ids, item_texts

def embedding_search(
    query_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    batch_size: int = 10,
    top_k: int = 100,
    distance: str = "l2",
) -> Iterable[Tuple[torch.Tensor, int]]:
    batch_offset = 0
    for batch in tqdm(batchify_slicing(query_embeddings, batch_size)):
        if distance == "l2":
            similarity = -torch.cdist(batch, item_embeddings, p=2)
        elif distance == "ip":
            similarity = batch @ item_embeddings.T
        
        top_k_items = torch.topk(similarity, top_k, dim=1).indices.cpu()
        yield (top_k_items, batch_offset)
        batch_offset += batch.shape[0]

def create_item_graph_with_item_embedding_search(
    encoded_items_path: str,
    output_path: str,
    device: str = "cuda",
    dtype: Union[torch.dtype, str] = "fp32",
    batch_size: int = 100,
    top_k: int = 100,
    distance: str = "l2",
) -> None:
    """
    Args:
        encoded_items_path (str): The path to the encoded items used to find
            the nearest neighbors.
        output_path (str): The path to the output jsonl file.
        device (str, optional): The device to use for the computation.
            Defaults to "cuda".
        dtype (Union[torch.dtype, str], optional): The dtype to use for the
            computation. Defaults to "fp32".
        batch_size (int, optional): The batch size to use for the computation.
            Defaults to 100.
        top_k (int, optional): The number of nearest neighbors to find.
            Defaults to 100.
        distance (str, optional): The distance metric to use. Options are
            "l2" or "ip". Defaults to "l2".
    """
    start = time.time()

    if isinstance(dtype, str):
        dtype = dtype_lookup(dtype)

    item_embeddings, item_ids, _ = get_embeddings(encoded_items_path)
    item_embeddings = item_embeddings.to(device, dtype=dtype)

    t1 = time.time()
    print(f"Loaded {len(item_ids)} embeddings to {device} in {t1 - start:.5f} seconds.")

    top_indices_iter = embedding_search(
        item_embeddings,
        item_embeddings,
        batch_size=batch_size,
        top_k=top_k,
        distance=distance,
    )

    t2 = time.time()
    print(f"Initialized embedding search iterator in {t2 - t1:.5f} seconds.")

    with JsonlWriter(output_path) as writer:
        for top_indices, offset in top_indices_iter:
            for i, item_id in enumerate(
                item_ids[offset : offset + top_indices.shape[0]]
            ):
                neighbor_indices = top_indices[i]

                neighbors = [
                    item_ids[neighbor_index]
                    for neighbor_index in neighbor_indices
                    if neighbor_index != i
                ]

                writer.write(
                    {
                        "item_id": item_id,
                        "neighbors": neighbors,
                    }
                )
    
    end = time.time()
    print(f"Total time to construct neighbor graph: {end - start:.5f} seconds.")

if __name__ == "__main__":
    fire.Fire(create_item_graph_with_item_embedding_search)
# Model access

- Selected model: UI-Mate-9B Q6_K; use the [Kaggle pilot](kaggle.md) until Anant's RTX 5080 endpoint is ready.
- `--model ui-mate` adapts native UI-Mate actions; `--model http` remains the generic JSON baseline for other models.
- Both profiles require a vision-capable server accepting `POST <base_url>/chat/completions` with inline base64 images ([vLLM contract](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)).
- Anant supplies the server address, exact served model ID, and API key; the compatible model/runtime and 16 GB fit still need testing.
- Use HTTPS or an SSH tunnel; HTTP is accepted only on localhost, and redirects are blocked.
- A tunnel can forward local port 8000 to Anant's server: `ssh -N -L 8000:127.0.0.1:8000 USER@ANANT_HOST` (requires his SSH setup).
- Set these PowerShell variables in the terminal running Vembody; values below are placeholders, not working credentials.

```powershell
$env:VEMBODY_ANANT_BASE_URL = "http://127.0.0.1:8000/v1" # Through the SSH tunnel.
$env:VEMBODY_ANANT_MODEL = "SERVED_MODEL_ID"
$env:VEMBODY_ANANT_API_KEY = "SERVER_API_KEY"
python -m vembody run --task tasks/benchmark/development/t1-01.yaml --model ui-mate --model-config configs/models/anant.yaml
```

- Paid hosting remains optional; select a service with compatible vision requests and fill in its credentials only if Kaggle is insufficient.

```powershell
$env:VEMBODY_HOSTED_BASE_URL = "https://HOST/v1"
$env:VEMBODY_HOSTED_MODEL = "HOSTED_MODEL_ID"
$env:VEMBODY_HOSTED_API_KEY = "HOSTED_API_KEY"
python -m vembody run --task tasks/benchmark/development/t1-01.yaml --model http --model-config configs/models/hosted.yaml
```

- For repeated trials, use `eval --tasks tasks/benchmark/development/t1-01.yaml --trials 3 --name UNIQUE_NAME` with the same model/profile flags; Kaggle starts with one pilot trial.
- Switching profiles is manual; failures do not retry, select another model, or automatically spend money.
- Requests contain the task instruction and current screenshot, not hidden board state, labels, or history; actions still pass strict parsing and validation.
- Keys stay in environment variables; experiment manifests record resolved endpoint/model/settings and prompt version without secret values.
- This configures inference only; training-job submission, host installation, and fine-tuning remain separate work.

# OpenShift CLI Tools Container

This container image provides OpenShift CLI tools and utilities needed for cluster credential management in the recommendation system workflow.

## Purpose

This specialized image is used by the `fetch_cluster_credentials` component in the Kubeflow pipeline to:
- Authenticate with the OpenShift cluster
- Retrieve user tokens and cluster information
- Get Model Registry service endpoints
- Extract routing information for external services

## Contents

### Tools Included
- **OpenShift CLI (`oc`)** - Latest version from OpenShift mirror
- **jq** - JSON processor for parsing API responses
- **curl** - For downloading the OpenShift CLI
- **tar** - For extracting downloaded archives

### Python Base
- **Python 3.11-slim** - Minimal Python runtime
- **model_registry** - Python package for model registry operations (installed via pip at runtime)

## Usage in Pipeline

This image is used by the `fetch_cluster_credentials()` function in `train-workflow.py`

## Building the Image

```bash
# From the oc-tools/ directory
podman build --platform linux/amd64 -t quay.io/ecosystem-appeng/model-registry .  
# Push to registry
podman push quay.io/rh-ee-ofridman/model-registry-python-oc
```

## Why Separate from Base Image?

This image is kept separate from the main `BASE_IMAGE` because:

1. **Security Isolation** - OpenShift CLI tools have cluster access privileges
2. **Image Size** - ML workloads don't need cluster management tools
3. **Separation of Concerns** - Infrastructure operations vs. ML operations
4. **Maintenance** - Can update OC tools independently of ML dependencies

## Environment Variables

The component using this image expects these environment variables:
- `MODEL_REGISTRY_NAMESPACE` - Namespace where model registry is deployed
- `MODEL_REGISTRY_CONTAINER` - Name of the model registry service

## Dependencies

This image requires the pod to have:
- ServiceAccount with cluster read permissions
- Access to the OpenShift API server
- Network connectivity to model registry services

## Related Components

- `registry_model_to_model_registry()` - Consumes the credentials from this component
- `train_model()` - Provides model artifacts to be registered
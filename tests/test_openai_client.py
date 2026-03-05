from core.clients.openai_client import _is_azure_openai_base_url


def test_is_azure_openai_base_url_accepts_openai_azure_domain():
    assert _is_azure_openai_base_url("https://myres.openai.azure.com/openai/v1/")


def test_is_azure_openai_base_url_accepts_services_ai_azure_domain():
    assert _is_azure_openai_base_url("https://xyz-eastus2.services.ai.azure.com/openai/v1/")


def test_is_azure_openai_base_url_accepts_services_ai_without_scheme():
    assert _is_azure_openai_base_url("xyz-eastus2.services.ai.azure.com/openai/v1/")


def test_is_azure_openai_base_url_accepts_quoted_url():
    assert _is_azure_openai_base_url('"https://xyz-eastus2.services.ai.azure.com/openai/v1/"')


def test_is_azure_openai_base_url_rejects_non_azure_domain():
    assert not _is_azure_openai_base_url("https://api.openai.com/v1/")

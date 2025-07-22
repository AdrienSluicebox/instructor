import pytest
import os
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("mode", [instructor.Mode.XAI_JSON, instructor.Mode.XAI_TOOLS])
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_xai_async_from_provider(mode):
    """Test xAI async client using from_provider with different modes"""
    client = instructor.from_provider("xai/grok-3-mini", mode=mode, async_client=True)

    user = await client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25


@pytest.mark.parametrize("mode", [instructor.Mode.XAI_JSON, instructor.Mode.XAI_TOOLS])
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_sync_from_provider(mode):
    """Test xAI sync client using from_provider with different modes"""
    client = instructor.from_provider("xai/grok-3-mini", mode=mode)

    user = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_xai_json_mode_raw_response_async():
    """Test that XAI_JSON mode includes _raw_response attribute in async calls"""
    client = instructor.from_provider("xai/grok-3-mini", mode=instructor.Mode.XAI_JSON, async_client=True)

    user = await client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Alice is 30 years old.",
            },
        ],
    )

    # Verify the parsed response works as expected
    assert isinstance(user, User)
    assert user.name == "Alice"
    assert user.age == 30
    
    # Verify that _raw_response attribute exists
    assert hasattr(user, "_raw_response"), "Parsed response should have _raw_response attribute"
    
    # Verify that _raw_response is not None and contains data
    assert user._raw_response is not None, "_raw_response should not be None"


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_json_mode_raw_response_sync():
    """Test that XAI_JSON mode includes _raw_response attribute in sync calls"""
    client = instructor.from_provider("xai/grok-3-mini", mode=instructor.Mode.XAI_JSON)

    user = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.", 
            },
            {
                "role": "user",
                "content": "Extract: Bob is 28 years old.",
            },
        ],
    )

    # Verify the parsed response works as expected
    assert isinstance(user, User)
    assert user.name == "Bob"
    assert user.age == 28
    
    # Verify that _raw_response attribute exists
    assert hasattr(user, "_raw_response"), "Parsed response should have _raw_response attribute"
    
    # Verify that _raw_response is not None and contains data
    assert user._raw_response is not None, "_raw_response should not be None"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_xai_json_mode_create_with_completion_async():
    """Test that XAI_JSON mode supports create_with_completion in async calls"""
    client = instructor.from_provider("xai/grok-3-mini", mode=instructor.Mode.XAI_JSON, async_client=True)

    user, raw_response = await client.chat.completions.create_with_completion(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Dana is 32 years old.",
            },
        ],
    )

    # Verify the parsed response works as expected
    assert isinstance(user, User)
    assert user.name == "Dana"
    assert user.age == 32
    
    # Verify that raw response is returned
    assert raw_response is not None, "Raw response should not be None"
    
    # Verify that _raw_response is also set on the model
    assert hasattr(user, "_raw_response"), "Parsed response should have _raw_response attribute"
    assert user._raw_response is raw_response, "_raw_response should match the returned raw response"


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_json_mode_create_with_completion_sync():
    """Test that XAI_JSON mode supports create_with_completion in sync calls"""
    client = instructor.from_provider("xai/grok-3-mini", mode=instructor.Mode.XAI_JSON)

    user, raw_response = client.chat.completions.create_with_completion(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Eve is 26 years old.",
            },
        ],
    )

    # Verify the parsed response works as expected
    assert isinstance(user, User)
    assert user.name == "Eve"
    assert user.age == 26
    
    # Verify that raw response is returned
    assert raw_response is not None, "Raw response should not be None"
    
    # Verify that _raw_response is also set on the model
    assert hasattr(user, "_raw_response"), "Parsed response should have _raw_response attribute"
    assert user._raw_response is raw_response, "_raw_response should match the returned raw response"


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_tools_mode_raw_response():
    """Test that XAI_TOOLS mode also includes _raw_response attribute"""
    client = instructor.from_provider("xai/grok-3-mini", mode=instructor.Mode.XAI_TOOLS)

    user = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user", 
                "content": "Extract: Charlie is 35 years old.",
            },
        ],
    )

    # Verify the parsed response works as expected
    assert isinstance(user, User)
    assert user.name == "Charlie"
    assert user.age == 35
    
    # Verify that _raw_response attribute exists for XAI_TOOLS mode too
    assert hasattr(user, "_raw_response"), "XAI_TOOLS mode should also have _raw_response attribute"
    assert user._raw_response is not None, "_raw_response should not be None"


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_tools_mode_create_with_completion():
    """Test that XAI_TOOLS mode also supports create_with_completion"""
    client = instructor.from_provider("xai/grok-3-mini", mode=instructor.Mode.XAI_TOOLS)

    user, raw_response = client.chat.completions.create_with_completion(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Frank is 29 years old.",
            },
        ],
    )

    # Verify the parsed response works as expected
    assert isinstance(user, User)
    assert user.name == "Frank"
    assert user.age == 29
    
    # Verify that raw response is returned
    assert raw_response is not None, "Raw response should not be None"
    
    # Verify that _raw_response is also set on the model
    assert hasattr(user, "_raw_response"), "Parsed response should have _raw_response attribute"
    assert user._raw_response is raw_response, "_raw_response should match the returned raw response"

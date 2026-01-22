"""
LangChain wrapper for custom LLM callers
Allows our LLM factory callers to work with LangChain agents
"""
from typing import Any, List, Optional, Iterator, AsyncIterator
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation, GenerationChunk
from langchain_core.runnables.config import run_in_executor
from src.llm_factory import BaseLLMCaller


class LangChainLLMWrapper(LLM):
    """Wrapper to make our LLM callers compatible with LangChain"""
    
    llm_caller: BaseLLMCaller
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM"""
        return f"custom_{self.llm_caller.__class__.__name__}"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM"""
        response = self.llm_caller.generate(prompt, **kwargs)

        # Ensure response is a string
        if isinstance(response, str):
            result = response
        elif hasattr(response, 'text'):
            result = response.text
        else:
            result = str(response)

        # Handle stop sequences if provided
        if stop and isinstance(result, str):
            for stop_seq in stop:
                if stop_seq in result:
                    result = result[:result.index(stop_seq)]
                    break

        return result
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts"""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"_generate called with {len(prompts)} prompts")
        
        generations = []
        for prompt in prompts:
            response = self.llm_caller.generate(prompt, **kwargs)
            
            # Ensure response is a string
            if isinstance(response, str):
                text = response
            elif hasattr(response, 'text'):
                text = response.text
            else:
                text = str(response)
            
            # Handle stop sequences if provided
            if stop and isinstance(text, str):
                for stop_seq in stop:
                    if stop_seq in text:
                        text = text[:text.index(stop_seq)]
                        break
            
            # Create Generation object - ensure text is a string
            if not isinstance(text, str):
                text = str(text)
            
            # Create Generation object with explicit text parameter
            # Use dict initialization to ensure proper Pydantic model creation
            generation = Generation(**{"text": text})
            
            # Validate that generation is actually a Generation object
            if not isinstance(generation, Generation):
                raise TypeError(f"Expected Generation object, got {type(generation)}")
            
            # Validate that generation has text attribute and it's accessible
            if not hasattr(generation, 'text'):
                raise AttributeError(f"Generation object missing 'text' attribute: {generation}")
            
            # Test accessing the text attribute
            try:
                _ = generation.text
            except Exception as e:
                raise AttributeError(f"Cannot access generation.text: {e}")
            
            generations.append([generation])
        
        # Validate and fix the result structure
        # Ensure all items in generations are actually Generation objects
        fixed_generations = []
        for gen_list in generations:
            fixed_gen_list = []
            for gen in gen_list:
                if isinstance(gen, Generation):
                    fixed_gen_list.append(gen)
                elif isinstance(gen, str):
                    # If it's a string, create a Generation object
                    fixed_gen_list.append(Generation(text=gen))
                else:
                    # Try to convert to string and create Generation
                    fixed_gen_list.append(Generation(text=str(gen)))
            fixed_generations.append(fixed_gen_list)
        
        result = LLMResult(generations=fixed_generations)
        
        # Final validation - ensure we have proper Generation objects
        if result.generations and len(result.generations) > 0:
            if len(result.generations[0]) > 0:
                first_gen = result.generations[0][0]
                logger.debug(f"_generate returning result with first_gen type: {type(first_gen)}")
                if not isinstance(first_gen, Generation):
                    # Last resort: recreate the entire result
                    logger.warning(
                        f"Generation object validation failed, recreating. "
                        f"Got type: {type(first_gen)}, value: {first_gen}"
                    )
                    # Recreate with proper Generation objects
                    new_generations = []
                    for gen_list in result.generations:
                        new_gen_list = []
                        for gen in gen_list:
                            if isinstance(gen, str):
                                new_gen_list.append(Generation(text=gen))
                            else:
                                new_gen_list.append(Generation(text=str(gen)))
                        new_generations.append(new_gen_list)
                    result = LLMResult(generations=new_generations)
        
        return result
    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate responses for multiple prompts"""
        # Use LangChain's run_in_executor which properly handles LLMResult serialization
        result = await run_in_executor(
            None,
            self._generate,
            prompts,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        
        # Critical: After executor, verify and fix Generation objects
        # The executor might have serialized/deserialized them incorrectly
        import logging
        logger = logging.getLogger(__name__)
        
        if result.generations:
            for i, gen_list in enumerate(result.generations):
                for j, gen in enumerate(gen_list):
                    if not isinstance(gen, Generation):
                        logger.warning(
                            f"_agenerate: Found non-Generation object at [{i}][{j}]: "
                            f"type={type(gen)}, value={gen}. Fixing..."
                        )
                        # Recreate as Generation object
                        if isinstance(gen, str):
                            result.generations[i][j] = Generation(text=gen)
                        else:
                            result.generations[i][j] = Generation(text=str(gen))
                    # Double-check the Generation object is valid
                    elif not hasattr(gen, 'text') or not isinstance(gen.text, str):
                        logger.warning(
                            f"_agenerate: Invalid Generation object at [{i}][{j}]: {gen}. Recreating..."
                        )
                        text = gen.text if hasattr(gen, 'text') else str(gen)
                        result.generations[i][j] = Generation(text=str(text))
        
        return result
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM output as GenerationChunk objects"""
        try:
            for chunk in self.llm_caller.stream(prompt, **kwargs):
                # Ensure chunk is a string
                if isinstance(chunk, str):
                    chunk_text = chunk
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                else:
                    chunk_text = str(chunk)

                # Handle stop sequences
                if stop and isinstance(chunk_text, str):
                    for stop_seq in stop:
                        if stop_seq in chunk_text:
                            chunk_text = chunk_text[:chunk_text.index(stop_seq)]
                            # Yield final chunk and return
                            if chunk_text:
                                yield GenerationChunk(text=chunk_text)
                            return

                # Yield as GenerationChunk (not Generation or string)
                if chunk_text:
                    yield GenerationChunk(text=chunk_text)
                    # Notify run manager if provided
                    if run_manager:
                        run_manager.on_llm_new_token(chunk_text)
        except Exception as e:
            # If streaming fails, fall back to non-streaming and yield as single chunk
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Streaming failed, falling back to non-streaming: {e}")
            try:
                result = self.llm_caller.generate(prompt, **kwargs)
                if isinstance(result, str):
                    result_text = result
                elif hasattr(result, 'text'):
                    result_text = result.text
                else:
                    result_text = str(result)
                
                # Handle stop sequences
                if stop and isinstance(result_text, str):
                    for stop_seq in stop:
                        if stop_seq in result_text:
                            result_text = result_text[:result_text.index(stop_seq)]
                            break
                
                # Yield as single chunk
                if result_text:
                    yield GenerationChunk(text=result_text)
                    if run_manager:
                        run_manager.on_llm_new_token(result_text)
            except Exception as fallback_error:
                logger.error(f"Fallback to non-streaming also failed: {fallback_error}")
                raise
    
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Async stream the LLM output as GenerationChunk objects"""
        # For now, use the synchronous stream in an executor
        # If the caller supports async streaming, we can enhance this later
        import asyncio
        loop = asyncio.get_event_loop()
        
        # Run the synchronous stream in an executor
        def sync_stream():
            return list(self._stream(prompt, stop, None, **kwargs))
        
        chunks = await loop.run_in_executor(None, sync_stream)
        
        for chunk in chunks:
            # Yield as GenerationChunk
            yield chunk
            # Notify async run manager if provided
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
    
    @property
    def _identifying_params(self) -> dict:
        """Get identifying parameters"""
        return {
            "model": self.llm_caller.model,
            "provider": self.llm_caller.__class__.__name__
        }


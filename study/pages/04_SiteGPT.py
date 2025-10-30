import streamlit as st

# meta tag 적용
st.set_page_config(
	page_title="SiteGPT",
	page_icon="🗒️",
)

st.title("SiteGPT")
st.markdown(
	"""
	웹사이트 안에 있는 컨텐츠에 대해 무엇이든 질문해보세요.
	
	사이드바의 입력칸에 원하는 컨텐츠가 있는 주소를 입력하세요.
	"""
)

with st.sidebar:
	url = st.text_input(
		"Write down a URL",
		placeholder="https://www.example.com",
	)


exam1 = '''
# 10.1 AsyncChromiumLoader

from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
html2text_transformer = Html2TextTransformer()

if url:
	async def load_documents():
		loader = AsyncChromiumLoader([url])
		docs = await loader.aload()
		return docs
	
	try:
		docs = asyncio.run(load_documents())
		st.write(docs)
		transformed = html2text_transformer.transform_documents(docs)
		st.write(transformed)
	except Exception as e:
		st.error(f"문서 로드 중 오류가 발생했습니다: {str(e)}")
		st.info("브라우저 로더가 실패했습니다. 다른 방법을 시도해보세요.")
'''


#10.2 Sitemap loader
from langchain.document_loaders import SitemapLoader
from fake_useragent import UserAgent

ua = UserAgent()

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
	try:
		loader = SitemapLoader(url)
		loader.requests_per_second = 5
		loader.headers = {'User-Agent': ua.random}
		docs = loader.load()
		return docs
	except Exception as e:
		st.error(f"Error loading sitemap : {e}")
		return []

if url:
	# if ".xml" not in url:
	#	with st.sidebar:
	#		st.error("Sitemap URL 을 적어주세요.")
	# else:
		docs = load_website(url)
		st.write(docs)
mod auth;
mod catalog;
mod mantle;

use std::path::PathBuf;
use std::sync::Arc;

use codex_api::Provider;
use codex_api::SharedAuthProvider;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_4_MODEL_ID;
use codex_model_provider_info::ModelProviderAwsAuthInfo;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::OPENAI_PROJECT_HEADER;
use codex_models_manager::manager::SharedModelsManager;
use codex_models_manager::manager::StaticModelsManager;
use codex_protocol::account::ProviderAccount;
use codex_protocol::error::Result;
use codex_protocol::openai_models::ModelsResponse;

use crate::provider::ModelProvider;
use crate::provider::ProviderAccountResult;
use crate::provider::ProviderAccountState;
use crate::provider::ProviderCapabilities;
use auth::resolve_provider_auth;
pub(crate) use catalog::static_model_catalog;
use catalog::with_default_only_service_tier;
use mantle::runtime_base_url;

/// Runtime provider for Amazon Bedrock's OpenAI-compatible Mantle endpoint.
#[derive(Clone, Debug)]
pub(crate) struct AmazonBedrockModelProvider {
    pub(crate) info: ModelProviderInfo,
    pub(crate) aws: ModelProviderAwsAuthInfo,
}

impl AmazonBedrockModelProvider {
    pub(crate) fn new(provider_info: ModelProviderInfo) -> Self {
        let aws = provider_info
            .aws
            .clone()
            .unwrap_or(ModelProviderAwsAuthInfo {
                profile: None,
                region: None,
                project: None,
            });
        Self {
            info: provider_info,
            aws,
        }
    }

    fn project_id(&self) -> Option<&str> {
        self.aws
            .project
            .as_deref()
            .map(str::trim)
            .filter(|project| !project.is_empty())
    }

    fn build_provider_info(&self) -> ModelProviderInfo {
        let info = self.info.clone();
        match self.project_id() {
            Some(project) => info.with_http_header(OPENAI_PROJECT_HEADER, project),
            None => info,
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for AmazonBedrockModelProvider {
    fn info(&self) -> &ModelProviderInfo {
        &self.info
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            namespace_tools: true,
            image_generation: false,
            web_search: false,
        }
    }

    fn approval_review_preferred_model(&self) -> &'static str {
        AMAZON_BEDROCK_GPT_5_4_MODEL_ID
    }

    fn memory_extraction_preferred_model(&self) -> &'static str {
        AMAZON_BEDROCK_GPT_5_4_MODEL_ID
    }

    fn memory_consolidation_preferred_model(&self) -> &'static str {
        AMAZON_BEDROCK_GPT_5_4_MODEL_ID
    }

    fn auth_manager(&self) -> Option<Arc<AuthManager>> {
        None
    }

    async fn auth(&self) -> Option<CodexAuth> {
        None
    }

    fn account_state(&self) -> ProviderAccountResult {
        Ok(ProviderAccountState {
            account: Some(ProviderAccount::AmazonBedrock),
            requires_openai_auth: false,
        })
    }

    async fn api_provider(&self) -> Result<Provider> {
        let mut info = self.build_provider_info();
        info.base_url = Some(runtime_base_url(&self.aws).await?);
        info.to_api_provider(/*auth_mode*/ None)
    }

    async fn runtime_base_url(&self) -> Result<Option<String>> {
        Ok(Some(runtime_base_url(&self.aws).await?))
    }

    async fn api_auth(&self) -> Result<SharedAuthProvider> {
        resolve_provider_auth(&self.aws).await
    }

    fn models_manager(
        &self,
        _codex_home: PathBuf,
        config_model_catalog: Option<ModelsResponse>,
    ) -> SharedModelsManager {
        Arc::new(StaticModelsManager::new(
            /*auth_manager*/ None,
            config_model_catalog.map_or_else(static_model_catalog, with_default_only_service_tier),
        ))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn api_provider_for_bedrock_bearer_token_uses_configured_region_endpoint() {
        let region = "eu-central-1";
        let mut api_provider_info =
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None);
        api_provider_info.base_url = Some(mantle::base_url(region).expect("supported region"));
        let api_provider = api_provider_info
            .to_api_provider(/*auth_mode*/ None)
            .expect("api provider should build");

        assert_eq!(
            api_provider.base_url,
            "https://bedrock-mantle.eu-central-1.api.aws/openai/v1"
        );
    }

    #[test]
    fn capabilities_disable_unsupported_hosted_tools() {
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
        );

        assert_eq!(
            provider.capabilities(),
            ProviderCapabilities {
                namespace_tools: true,
                image_generation: false,
                web_search: false,
            }
        );
    }

    #[test]
    fn approval_review_preferred_model_uses_bedrock_gpt_5_4() {
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
        );

        assert_eq!(
            provider.approval_review_preferred_model(),
            AMAZON_BEDROCK_GPT_5_4_MODEL_ID
        );
    }

    #[test]
    fn project_header_injected_when_configured() {
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(Some(ModelProviderAwsAuthInfo {
                profile: None,
                region: None,
                project: Some("proj_abc123".to_string()),
            })),
        );

        let api_provider = provider
            .build_provider_info()
            .to_api_provider(/*auth_mode*/ None)
            .expect("api provider should build");

        assert_eq!(
            api_provider
                .headers
                .get(OPENAI_PROJECT_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some("proj_abc123")
        );
        assert!(
            api_provider
                .headers
                .contains_key("x-amzn-mantle-client-agent")
        );
    }

    #[test]
    fn project_header_absent_when_blank() {
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(Some(ModelProviderAwsAuthInfo {
                profile: None,
                region: None,
                project: Some("   ".to_string()),
            })),
        );

        let api_provider = provider
            .build_provider_info()
            .to_api_provider(/*auth_mode*/ None)
            .expect("api provider should build");

        assert!(!api_provider.headers.contains_key(OPENAI_PROJECT_HEADER));
    }

    #[test]
    fn project_header_absent_when_not_configured() {
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
        );

        let api_provider = provider
            .build_provider_info()
            .to_api_provider(/*auth_mode*/ None)
            .expect("api provider should build");

        assert!(!api_provider.headers.contains_key(OPENAI_PROJECT_HEADER));
    }
}

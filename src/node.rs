
pub enum DomainType {
    Discrete(Vec<String>)
}

pub struct Node {
    pub domain: DomainType,
    pub label: String
}



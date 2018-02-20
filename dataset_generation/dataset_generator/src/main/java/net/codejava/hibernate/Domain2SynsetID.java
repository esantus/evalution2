package net.codejava.hibernate;

import java.io.Serializable;
import java.util.Objects;

import javax.persistence.Column;
import javax.persistence.Embeddable;

@Embeddable
public class Domain2SynsetID implements Serializable {
	private static final long serialVersionUID = 1L;
	@Column(name="synset_id")
	Long synset_id;
	
	@Column(name="domain_Id")
	Long domain_id;

	
	public Domain2SynsetID() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public Domain2SynsetID(Long synset_id, Long domain_id) {
		super();
		this.synset_id = synset_id;
		this.domain_id = domain_id;
	}

	public Long getSynset_id() {
		return synset_id;
	}


	public void setSynset_id(Long synset_id) {
		this.synset_id = synset_id;
	}


	public Long getDomain_id() {
		return domain_id;
	}


	public void setDomain_id(Long domain_id) {
		this.domain_id = domain_id;
	}

	@Override
	public String toString() {
		return "Domain2SynsetID [synset_id=" + synset_id + ", domain_id=" + domain_id + "]";
	}

	public boolean equals(Object o){
		if (this == o) return true;
		if (!(o instanceof Domain2SynsetID)) return false;
		Domain2SynsetID that = (Domain2SynsetID) o;
		return Objects.equals(getSynset_id(), that.getSynset_id()) && 
				Objects.equals(getDomain_id(), that.getDomain_id());
	}
	
	public int hashCode(){
		return Objects.hash(getSynset_id(), getDomain_id());
	}
}

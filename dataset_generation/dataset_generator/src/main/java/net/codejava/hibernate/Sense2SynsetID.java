package net.codejava.hibernate;

import java.io.Serializable;
import java.util.Objects;

import javax.persistence.Column;
import javax.persistence.Embeddable;

@Embeddable
public class Sense2SynsetID implements Serializable {
	private static final long serialVersionUID = 1L;
	@Column(name="synset_id")
	Long synset_id;
	
	@Column(name="synsetSense_id")
	Long synsetSense_id;
	
	public Sense2SynsetID() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public Sense2SynsetID(Long synset_id, Long synsetSense_id) {
		super();
		this.synset_id = synset_id;
		this.synsetSense_id = synsetSense_id;
	}

	public Long getSynset_id() {
		return synset_id;
	}

	public void setSynset_id(Long synset_id) {
		this.synset_id = synset_id;
	}

	public Long getSynsetSense_id() {
		return synsetSense_id;
	}

	public void setSynsetSense_id(Long synsetSenses_id) {
		this.synsetSense_id = synsetSenses_id;
	}

	@Override
	public String toString() {
		return "Sense2SynsetID [synset_id=" + synset_id + ", synsetSenses_id=" + synsetSense_id + "]";
	}

	public boolean equals(Object o){
		if (this == o) return true;
		if (!(o instanceof Sense2SynsetID)) return false;
		Sense2SynsetID that = (Sense2SynsetID) o;
		return Objects.equals(getSynset_id(), that.getSynset_id()) && 
				Objects.equals(getSynsetSense_id(), that.getSynsetSense_id());
	}
	
	public int hashCode(){
		return Objects.hash(getSynset_id(), getSynsetSense_id());
	}
}

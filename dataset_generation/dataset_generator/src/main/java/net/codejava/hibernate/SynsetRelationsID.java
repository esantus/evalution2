package net.codejava.hibernate;

import java.io.Serializable;
import java.util.Objects;

import javax.persistence.Column;
import javax.persistence.Embeddable;

@Embeddable
public class SynsetRelationsID implements Serializable {
	private static final long serialVersionUID = 1L;
	@Column(name="sourceSynset_id")
	Long sourceSynset_id;
	
	@Column(name="targetSynset_id")
	Long targetSynset_id;
	
	@Column(name="relation_id")
	Long relation_id;

	
	public SynsetRelationsID() {
		super();
		// TODO Auto-generated constructor stub
	}

	
	public SynsetRelationsID(Long sourceSynset_id, Long targetSynset_id, Long relation_id) {
		super();
		this.sourceSynset_id = sourceSynset_id;
		this.targetSynset_id = targetSynset_id;
		this.relation_id = relation_id;
	}


	public Long getSourceSynset_id() {
		return sourceSynset_id;
	}



	public void setSourceSynset_id(Long sourceSynset_id) {
		this.sourceSynset_id = sourceSynset_id;
	}



	public Long getTargetSynset_id() {
		return targetSynset_id;
	}



	public void setTargetSynset_id(Long targetSynset_id) {
		this.targetSynset_id = targetSynset_id;
	}



	public Long getRelation_id() {
		return relation_id;
	}



	public void setRelation_id(Long relation_id) {
		this.relation_id = relation_id;
	}
	
	@Override
	public String toString() {
		return "SynsetRelationsID [sourceSynset_id=" + sourceSynset_id + ", targetSynset_id=" + targetSynset_id
				+ ", relation_id=" + relation_id + "]";
	}


	public boolean equals(Object o){
		if (this == o) return true;
		if (!(o instanceof SynsetRelationsID)) return false;
		SynsetRelationsID that = (SynsetRelationsID) o;
		return Objects.equals(getSourceSynset_id(), that.getSourceSynset_id()) && 
				Objects.equals(getTargetSynset_id(), that.getTargetSynset_id()) && 
				Objects.equals(getRelation_id(), that.getRelation_id());
	}
	
	public int hashCode(){
		return Objects.hash(getSourceSynset_id(), getTargetSynset_id(), getRelation_id());
	}
}

package net.codejava.hibernate;

import javax.persistence.*;

@Entity
@Table(name="synsetRelations")
public class SynsetRelations{
	@EmbeddedId
	private SynsetRelationsID id; 
	private Double relationWeight;
	private String relationSurfaceText;
	
	public SynsetRelations() {
		super();
		// TODO Auto-generated constructor stub
	}

	public SynsetRelations(SynsetRelationsID id, Double relationWeight, String relationSurfaceText) {
		super();
		this.id = id;
		this.relationWeight = relationWeight;
		this.relationSurfaceText = relationSurfaceText;
	}

	public SynsetRelationsID getId() {
		return id;
	}

	public void setId(SynsetRelationsID id) {
		this.id = id;
	}

	public Double getRelationWeight() {
		return relationWeight;
	}

	public void setRelationWeight(Double relationWeight) {
		this.relationWeight = relationWeight;
	}

	public String getRelationSurfaceText() {
		return relationSurfaceText;
	}

	public void setRelationSurfaceText(String relationSurfaceText) {
		this.relationSurfaceText = relationSurfaceText;
	}

	@Override
	public String toString() {
		return "SynsetRelations [id=" + id + ", relationWeight=" + relationWeight + ", relationSurfaceText="
				+ relationSurfaceText + "]";
	}

}

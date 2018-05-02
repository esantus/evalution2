package net.codejava.hibernate;

import javax.persistence.*;

@Entity
@Table(name="domain2Synset")
public class Domain2Synset{
	@EmbeddedId
	private Domain2SynsetID id;
	private Double score;
	private Long source_id;
	
	public Domain2Synset() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Domain2Synset(Domain2SynsetID id, Double score, Long source_id) {
		super();
		this.id = id;
		this.score = score;
		this.source_id = source_id;
	}

	public Domain2SynsetID getId() {
		return id;
	}

	public void setId(Domain2SynsetID id) {
		this.id = id;
	}

	public Double getScore() {
		return score;
	}

	public void setScore(Double score) {
		this.score = score;
	}

	public Long getSource_id() {
		return source_id;
	}

	public void setSource_id(Long source_id) {
		this.source_id = source_id;
	}

	@Override
	public String toString() {
		return "Domain2Synset [id=" + id + ", score=" + score + ", source_id=" + source_id + "]";
	}
}

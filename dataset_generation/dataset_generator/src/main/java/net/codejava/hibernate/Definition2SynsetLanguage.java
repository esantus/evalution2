package net.codejava.hibernate;

import javax.persistence.*;

@Entity
@Table(name="definition2SynsetLanguage")
public class Definition2SynsetLanguage{
	@EmbeddedId
	private Definition2SynsetID id;
	private Long source_id;
		
	public Definition2SynsetLanguage() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Definition2SynsetLanguage(Definition2SynsetID id, Long source_id) {
		super();
		this.id = id;
		this.source_id = source_id;
	}

	public Definition2SynsetID getId() {
		return id;
	}

	public void setId(Definition2SynsetID id) {
		this.id = id;
	}

	public Long getSource_id() {
		return source_id;
	}

	public void setSource_id(Long source_id) {
		this.source_id = source_id;
	}

	@Override
	public String toString() {
		return "Definition2Synset [id=" + id + ", source_id=" + source_id + "]";
	}

}
